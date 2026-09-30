# terraform/variables.tf

variable "aws_region" {
  type        = string
  default     = "sa-east-1" # São Paulo region for regional compliance
  description = "AWS Region for deployment"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Target environment name"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.100.0.0/16"
  description = "CIDR block for the dedicated serving VPC"
}

variable "cluster_name" {
  type        = string
  default     = "credit-bureau-prod-cluster"
  description = "EKS cluster name"
}

variable "ecr_repository_name" {
  type        = string
  default     = "credit-decision-api"
  description = "ECR repository name for serving container"
}