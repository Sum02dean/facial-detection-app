import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import logging
import time
from credentials import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_REGION


logger = logging.getLogger(__name__)
MAX_GET_SIZE = 100

def do_batch_get(batch_keys):
    """
    Gets a batch of items from Amazon DynamoDB. Batches can contain keys from
    more than one table.

    When Amazon DynamoDB cannot process all items in a batch, a set of unprocessed
    keys is returned. This function uses an exponential backoff algorithm to retry
    getting the unprocessed keys until all are retrieved or the specified
    number of tries is reached.

    :param batch_keys: The set of keys to retrieve. A batch can contain at most 100
                       keys. Otherwise, Amazon DynamoDB returns an error.
    :return: The dictionary of retrieved items grouped under their respective
             table names.
    """
    tries = 0
    max_tries = 5
    sleepy_time = 1  # Start with 1 second of sleep, then exponentially increase.
    retrieved = {key: [] for key in batch_keys}
    while tries < max_tries:
        response = dynamodb.batch_get_item(RequestItems=batch_keys)

        # Collect any retrieved items and retry unprocessed keys.
        for key in response.get("Responses", []):
            retrieved[key] += response["Responses"][key]
        unprocessed = response["UnprocessedKeys"]
        if len(unprocessed) > 0:
            batch_keys = unprocessed
            unprocessed_count = sum(
                [len(batch_key["Keys"]) for batch_key in batch_keys.values()]
            )
            logger.info(
                "%s unprocessed keys returned. Sleep, then retry.", unprocessed_count
            )
            tries += 1
            if tries < max_tries:
                logger.info("Sleeping for %s seconds.", sleepy_time)
                time.sleep(sleepy_time)
                sleepy_time = min(sleepy_time * 2, 32)
        else:
            break

    return retrieved

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class DynamoDBTable:
    """Encapsulates an Amazon DynamoDB table of image data."""

    def __init__(self, dyn_resource):
        """
        :param dyn_resource: A Boto3 DynamoDB resource.
        """
        self.dyn_resource = dyn_resource
        self.table = None

    def exists(self, table_name):
        """
        Determines whether a table exists. As a side effect, stores the table in
        a member variable.

        :param table_name: The name of the table to check.
        :return: True when the table exists; otherwise, False.
        """
        try:
            table = self.dyn_resource.Table(table_name)
            table.load()
            exists = True
        except ClientError as err:
            if err.response["Error"]["Code"] == "ResourceNotFoundException":
                exists = False
            else:
                logger.error(
                    "Couldn't check for existence of %s. Here's why: %s: %s",
                    table_name,
                    err.response["Error"]["Code"],
                    err.response["Error"]["Message"],
                )
                raise
        else:
            self.table = table
        return exists
    
    def create_table(self, table_name):
        """
        Creates an Amazon DynamoDB table that can be used to store image data.

        :param table_name: The name of the table to create.
        :return: The newly created table.
        """
        try:
            self.table = self.dyn_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": "train_set", "KeyType": "HASH"},  # Partition key
                    {"AttributeName": "image_name", "KeyType": "RANGE"},  # Sort key
                ],
                AttributeDefinitions=[
                    {"AttributeName": "train_set", "AttributeType": "N"},
                    {"AttributeName": "image_name", "AttributeType": "S"},
                ],
                ProvisionedThroughput={
                    "ReadCapacityUnits": 10,
                    "WriteCapacityUnits": 10,
                },
            )
            self.table.wait_until_exists()
        except ClientError as err:
            logger.error(
                "Couldn't create table %s. Here's why: %s: %s",
                table_name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
        else:
            return self.table

    def delete_table(self):
        """
        Deletes the table.
        """
        try:
            self.table.delete()
            self.table = None
        except ClientError as err:
            logger.error(
                "Couldn't delete table. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

    def add_entry(self, json_file):
        """
        Adds an entry to the table.

        :param image_name: The path of the image.
        :param num_faces: The number of faces in the image.
        :param faces: the coordinates of the faces.
        :param rating: Boolean, should be in train set.
        """
        try:
            self.table.put_item(
                Item={
                    "image_name": json_file["image_name"],
                    "num_faces": json_file["num_faces"],
                    "faces": json_file["faces"],
                    "train_set": json_file["train_set"],
                }
            )
        except ClientError as err:
            logger.error(
                "Couldn't add entry %s to table %s. Here's why: %s: %s",
                json_file["image_name"],
                self.table.name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

    def write_batch(self, entries):
        """
        Fills an Amazon DynamoDB table with the specified data, using the Boto3
        Table.batch_writer() function to put the items in the table.
        Inside the context manager, Table.batch_writer builds a list of
        requests. On exiting the context manager, Table.batch_writer starts sending
        batches of write requests to Amazon DynamoDB and automatically
        handles chunking, buffering, and retrying.

        :param movies: The data to put in the table. Each item must contain at least
                       the keys required by the schema that was specified when the
                       table was created.
        """
        try:
            with self.table.batch_writer() as writer:
                for entry in entries:
                    writer.put_item(Item=entry)

        except ClientError as err:
            logger.error(
                "Couldn't load data into table %s. Here's why: %s: %s",
                self.table.name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

    def query_table(self, query_by="is_train", query=1):
        """
        Queries for observations based on criteria.

        :param query_by: Query criteria.
        :return: The list of entries that contain the key.
        """
        try:
            response = self.table.query(KeyConditionExpression=Key(query_by).eq(query))
        except ClientError as err:
            logger.error(
                "Couldn't query for entries with name %s. Here's why: %s: %s",
                query,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
        else:
            return response["Items"]

    def get_entry(self, image_name, is_train):
        """
        Gets entry from the table for a specific movie.

        :param image_name: The image_name of the entry.
        :param is_train: Boolean, in train set (1) or not (0).
        :return: The data about the requested entry.
        """
        try:
            response = self.table.get_item(Key={"image_name": image_name, "is_train": is_train})
        except ClientError as err:
            logger.error(
                "Couldn't get entry %s from table %s. Here's why: %s: %s",
                image_name,
                self.table.name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
        else:
            return response["Item"]
