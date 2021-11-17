# Databricks notebook source
# MAGIC %md
# MAGIC # HAR Data Setup
# MAGIC ## Mount S3 bucket
# MAGIC ### Unmount bucket if already exist

# COMMAND ----------

mount_name = "s3-mounted-input"
aws_bucket_name = "databricks-input-data"

# COMMAND ----------

dbutils.fs.unmount("/mnt/%s" % mount_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mount bucket

# COMMAND ----------

access_key = dbutils.secrets.get(scope = "aws", key = "aws-access-key")
secret_key = dbutils.secrets.get(scope = "aws", key = "aws-secret-key")
encoded_secret_key = secret_key.replace("/", "%2F")

dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)

# COMMAND ----------

display(dbutils.fs.ls("/mnt/%s" % mount_name))
