from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import subprocess
import re
from typing import List, Optional, Tuple

router = APIRouter(prefix="/api/network", tags=["network"])

_SYS_NET = "/sys/class/net"

# Virtual / non-edge interfaces to skip when picking primary Ethernet
_ETHER_SKIP_PREFIXES = (
    "docker",
    "br-",
    "virbr",
    "veth",
    "wlan",
    "wl",
    "p2p",
    "can",
    "tun",
    "tap",
    "dummy",
    "ifb",
    "l4tbr0",
)
_ETHER_SKIP_EXACT = frozenset({"lo", "l4tbr0"})


def _list_sys_net_ifaces() -> List[str]:
    try:
        return sorted(os.listdir(_SYS_NET))
    except OSError:
        return []


def _is_ethernet_candidate(name: str) -> bool:
    if name in _ETHER_SKIP_EXACT:
        return False
    if any(name.startswith(p) for p in _ETHER_SKIP_PREFIXES):
        return False
    if not (name.startswith("eth") or name.startswith("en")):
        return False
    return True


def _iface_has_device(name: str) -> bool:
    return os.path.isdir(os.path.join(_SYS_NET, name, "device"))


def detect_ethernet_interfaces() -> List[str]:
    """Names of likely wired Ethernet interfaces (eth*, en*), excluding common virtual ifaces."""
    names = [n for n in _list_sys_net_ifaces() if _is_ethernet_candidate(n)]
    with_dev = [n for n in names if _iface_has_device(n)]
    pool = with_dev if with_dev else names
    if "eth0" in pool:
        return ["eth0"] + [n for n in sorted(pool) if n != "eth0"]
    return sorted(pool)


def detect_primary_ethernet() -> Optional[str]:
    """Pick primary wired Ethernet (eth0 if present, else first sorted en*/eth*)."""
    ifaces = detect_ethernet_interfaces()
    return ifaces[0] if ifaces else None


def _interface_exists(name: str) -> bool:
    return os.path.isdir(os.path.join(_SYS_NET, name))


class SetInterfaceRequest(BaseModel):
    interface: str
    address: str


def _format_ipv4_list(pairs: List[Tuple[str, str]]) -> str:
    """Join IPv4/CIDR strings; skip loopback; stable order (non-APIPA first)."""
    out: List[str] = []
    apipa: List[str] = []
    for ip, pfx in pairs:
        if ip.startswith("127."):
            continue
        s = f"{ip}/{pfx}" if pfx else ip
        if ip.startswith("169.254."):
            apipa.append(s)
        else:
            out.append(s)
    ordered = out + apipa
    return ", ".join(ordered) if ordered else ""


def get_interface_address(interface: str) -> Optional[str]:
    """Get current IPv4 address(es) for a network interface (CIDR if known)."""
    if not interface:
        return None
    try:
        # IPv4 only avoids picking the wrong first inet among v4/v6 layout differences
        result = subprocess.run(
            ["ip", "-4", "addr", "show", "dev", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            pairs = re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)(?:/(\d+))?", result.stdout)
            if pairs:
                joined = _format_ipv4_list(pairs)
                return joined or None

        # Fallback to 'ifconfig' if 'ip' fails
        result = subprocess.run(
            ["ifconfig", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            pairs = re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if pairs:
                joined = _format_ipv4_list([(ip, "") for ip in pairs])
                return joined or None
    except Exception as e:
        print(f"[NETWORK] Error getting address for {interface}: {e}")

    return None


def set_interface_address(interface: str, address: str) -> bool:
    """Set IP address for a network interface."""
    try:
        # Parse address (may include CIDR notation like 192.168.1.100/24)
        if "/" in address:
            ip, prefix = address.split("/", 1)
        else:
            ip = address
            prefix = None
        
        # Use 'ip' command to set address
        # First, remove existing address if any
        subprocess.run(
            ["ip", "addr", "flush", "dev", interface],
            capture_output=True,
            timeout=5,
        )
        
        # Add new address
        if prefix:
            cmd = ["ip", "addr", "add", f"{ip}/{prefix}", "dev", interface]
        else:
            # Default to /24 if no prefix specified
            cmd = ["ip", "addr", "add", f"{ip}/24", "dev", interface]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # Bring interface up
            subprocess.run(
                ["ip", "link", "set", interface, "up"],
                capture_output=True,
                timeout=5,
            )
            return True
        else:
            print(f"[NETWORK] Failed to set address: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[NETWORK] Timeout setting address for {interface}")
        return False
    except Exception as e:
        print(f"[NETWORK] Error setting address for {interface}: {e}")
        return False


@router.get("/config")
async def get_network_config():
    """Get l4tbr0 and primary wired Ethernet address (iface name is auto-detected, e.g. enP8p1s0)."""
    try:
        l4tbr0 = get_interface_address("l4tbr0")
        primary_if = detect_primary_ethernet()
        primary_addr = get_interface_address(primary_if) if primary_if else None

        return {
            "ok": True,
            "l4tbr0": l4tbr0 or "",
            "primary_ether_if": primary_if or "",
            "primary_ether": primary_addr or "",
            # Deprecated: was hardcoded eth0; kept for any stale clients
            "eth0": primary_addr or "",
        }
    except Exception as e:
        print(f"[NETWORK] Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get network config: {str(e)}")


def _validate_set_interface(interface: str) -> None:
    """Raise HTTPException if interface is not allowed (l4tbr0 or existing sysfs iface)."""
    if interface == "l4tbr0":
        return
    if not re.match(r"^[a-zA-Z0-9._-]+$", interface) or len(interface) > 15:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interface name: {interface}",
        )
    if not _interface_exists(interface):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown interface: {interface}",
        )


@router.post("/set")
async def set_network_interface(request: SetInterfaceRequest):
    """Set IP address for a network interface."""
    # Do not lower-case: Linux names are case-sensitive (e.g. enP8p1s0 on Jetson).
    interface = request.interface.strip()
    address = request.address.strip()

    _validate_set_interface(interface)
    
    # Basic IP validation
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$"
    if not re.match(ip_pattern, address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid IP address format: {address}"
        )
    
    # Validate IP octets
    ip_part = address.split("/")[0]
    octets = ip_part.split(".")
    if len(octets) != 4 or not all(0 <= int(o) <= 255 for o in octets):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid IP address: {address}"
        )
    
    # Set the address
    success = set_interface_address(interface, address)
    
    if success:
        # Verify it was set
        new_address = get_interface_address(interface)
        return {
            "ok": True,
            "interface": interface,
            "address": new_address or address,
            "message": f"Successfully set {interface} to {address}",
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set {interface} to {address}. Check system logs."
        )
