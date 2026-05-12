locals {
  env       = terraform.workspace
  base_tags = {Owner = "team", Env = local.env}
  subnets   = [for az in var.azs : "subnet-${az}"]
}
