#!/bin/bash
# Setup script for M.2 SSD on Jetson Orin Nano
# This script will format and mount the M.2 SSD

set -e

DEVICE="/dev/nvme0n1"
MOUNT_POINT="/mnt/ssd"

echo "=========================================="
echo "M.2 SSD Setup Script"
echo "=========================================="
echo "Device: $DEVICE"
echo "Mount point: $MOUNT_POINT"
echo ""

# Check if device exists
if [ ! -b "$DEVICE" ]; then
    echo "ERROR: Device $DEVICE not found!"
    exit 1
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "WARNING: $MOUNT_POINT is already mounted!"
    echo "Unmounting..."
    sudo umount "$MOUNT_POINT"
fi

# Check if mount point exists, create if not
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point: $MOUNT_POINT"
    sudo mkdir -p "$MOUNT_POINT"
fi

# Check if device has partitions
PARTITIONS=$(lsblk -ln "$DEVICE" | grep -c part || true)
if [ "$PARTITIONS" -eq 0 ]; then
    echo ""
    echo "Creating partition table (GPT)..."
    sudo parted "$DEVICE" --script mklabel gpt
    
    echo "Creating partition (using full disk)..."
    sudo parted "$DEVICE" --script mkpart primary ext4 0% 100%
    
    echo "Waiting for partition to be recognized..."
    sleep 2
    
    PARTITION="${DEVICE}p1"
else
    echo "Partition(s) already exist, using first partition..."
    PARTITION="${DEVICE}p1"
fi

# Check if partition exists
if [ ! -b "$PARTITION" ]; then
    echo "ERROR: Partition $PARTITION not found!"
    exit 1
fi

echo ""
echo "Formatting partition as ext4..."
echo "WARNING: This will erase all data on $PARTITION!"
read -p "Press Enter to continue or Ctrl+C to cancel..."
sudo mkfs.ext4 -F "$PARTITION"

echo ""
echo "Mounting partition..."
sudo mount "$PARTITION" "$MOUNT_POINT"

# Set ownership to current user
echo "Setting ownership..."
sudo chown -R $USER:$USER "$MOUNT_POINT"

# Check if already in fstab
if ! grep -q "$PARTITION" /etc/fstab 2>/dev/null; then
    echo ""
    echo "Adding to /etc/fstab for automatic mounting..."
    UUID=$(sudo blkid -s UUID -o value "$PARTITION")
    echo "$PARTITION $MOUNT_POINT ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab
    echo "Added entry: UUID=$UUID -> $MOUNT_POINT"
else
    echo "Entry already exists in /etc/fstab"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Device: $PARTITION"
echo "Mount point: $MOUNT_POINT"
echo "Filesystem: ext4"
echo ""
df -h "$MOUNT_POINT"
echo ""
echo "You can now use: $MOUNT_POINT"
