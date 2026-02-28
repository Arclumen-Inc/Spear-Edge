from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import re
from typing import Optional

router = APIRouter(prefix="/api/network", tags=["network"])


class SetInterfaceRequest(BaseModel):
    interface: str
    address: str


def get_interface_address(interface: str) -> Optional[str]:
    """Get current IP address for a network interface."""
    try:
        # Use 'ip' command (preferred on modern Linux)
        result = subprocess.run(
            ["ip", "addr", "show", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse output: inet 192.168.1.100/24
            match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)(?:/(\d+))?", result.stdout)
            if match:
                ip = match.group(1)
                prefix = match.group(2)
                if prefix:
                    return f"{ip}/{prefix}"
                return ip
        
        # Fallback to 'ifconfig' if 'ip' fails
        result = subprocess.run(
            ["ifconfig", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse output: inet 192.168.1.100 netmask 255.255.255.0
            match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
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
    """Get current network configuration for l4tbr0 and eth0."""
    try:
        l4tbr0 = get_interface_address("l4tbr0")
        eth0 = get_interface_address("eth0")
        
        return {
            "ok": True,
            "l4tbr0": l4tbr0 or "",
            "eth0": eth0 or "",
        }
    except Exception as e:
        print(f"[NETWORK] Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get network config: {str(e)}")


@router.post("/set")
async def set_network_interface(request: SetInterfaceRequest):
    """Set IP address for a network interface."""
    interface = request.interface.strip().lower()
    address = request.address.strip()
    
    # Validate interface name
    if interface not in ("l4tbr0", "eth0"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interface: {interface}. Must be 'l4tbr0' or 'eth0'"
        )
    
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
