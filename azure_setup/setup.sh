#!/bin/bash

# Variables
RG="ccopenenclave"
LOCATION="eastus"
VM_NAME="accvm"
VM_SIZE="Standard_DC1s_v2"
ADMIN_USERNAME="user123"
ADMIN_PASSWORD="ReplaceWithSecurePassword123!"

# Create Resource Group
az group create --name $RG --location $LOCATION

# Create VNet and Subnet
az network vnet create \
    --resource-group $RG \
    --name "${VM_NAME}-vnet" \
    --subnet-name "${VM_NAME}-subnet"

# Create Public IP
az network public-ip create \
    --resource-group $RG \
    --name "${VM_NAME}-pip"

# Create NIC
az network nic create \
    --resource-group $RG \
    --name "${VM_NAME}-nic" \
    --vnet-name "${VM_NAME}-vnet" \
    --subnet "${VM_NAME}-subnet" \
    --public-ip-address "${VM_NAME}-pip"

# Create SGX-enabled VM
az vm create \
    --resource-group $RG \
    --name $VM_NAME \
    --location $LOCATION \
    --size $VM_SIZE \
    --image Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest \
    --admin-username $ADMIN_USERNAME \
    --admin-password $ADMIN_PASSWORD \
    --authentication-type password \
    --nics "${VM_NAME}-nic" \
    --storage-sku Premium_LRS \
    --os-disk-size-gb 30 \
    --enable-agent \
    --no-wait

echo "VM creation initiated. Check Azure Portal for progress."
