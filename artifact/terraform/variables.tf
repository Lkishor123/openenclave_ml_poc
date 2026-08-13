variable "env" {
  type    = string
  default = "dev"
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "resource_group_name" {
  type    = string
  default = "ccopenenclave"
}

variable "vm_name" {
  type    = string
  default = "accvm"
}

variable "vm_size" {
  type    = string
  default = "Standard_DC1s_v2"
}

variable "admin_username" {
  type    = string
  default = "user123"
}

variable "ssh_public_key" {
  type = string
}
