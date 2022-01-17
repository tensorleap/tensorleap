
module "project" {
  source   = "./project"
  name     = "${local.client_name}-dev-project"
  owners    = local.owners
  editors  = local.editors
}


