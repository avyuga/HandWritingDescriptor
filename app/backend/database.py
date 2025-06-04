import logging
from contextlib import asynccontextmanager
from pymongo import AsyncMongoClient
from bson.codec_options import CodecOptions
from uuid import UUID


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URL = "mongodb://mongo-db:27017"
DATABASE_NAME = "handwriting_descriptor"
MAX_POOL_SIZE = 100
MIN_POOL_SIZE = 10

# Add these new settings
MONGO_USERNAME = "admin"  
MONGO_PASSWORD = "admin"  

# Update the connection URL to include credentials
MONGO_URL = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@mongo-db:27017/{DATABASE_NAME}?authSource=admin"

# Schema definitions
SCHEMAS = {
    "predictions": {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["user_id", "request_id", "image_id", "created_at"],
                "properties": {
                    "user_id": {"bsonType": "string"},
                    "request_id": {"bsonType": "string"},
                    "image_id": {
                        "bsonType": ["string", "binData"],  # Allow both string and UUID
                        "description": "must be a string or UUID"
                    },
                    "created_at": {"bsonType": "date"},
                    "processing_time": {"bsonType": "double"},
                    "detections": {
                        "bsonType": "array",
                        "items": {
                            "bsonType": "object",
                            "required": ["bbox", "text", "score"],
                            "properties": {
                                "bbox": {"bsonType": "array", "items": {"bsonType": "double"}},
                                "text": {"bsonType": "string"},
                                "score": {"bsonType": "double"}
                            }
                        }
                    },
                    "user_rating": {
                        "bsonType": ["int", "null"],  # Allow both integer and null
                        "description": "must be an integer or null"
                    },
                    "user_transcription": {
                        "bsonType": ["string", "null"],  # Allow both string and null
                        "description": "must be a string or null"
                    }
                }
            }
        }
    }
}



class Database:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        """Create database connection with connection pooling."""
        if not self.client:
            try:
                # Configure codec options to handle UUIDs
                codec_options = CodecOptions(uuid_representation=4)  # 4 is for UUID_SUBTYPE
                
                self.client = AsyncMongoClient(
                    MONGO_URL,
                    maxPoolSize=MAX_POOL_SIZE,
                    minPoolSize=MIN_POOL_SIZE,
                    serverSelectionTimeoutMS=2000,
                    connectTimeoutMS=2000,
                    waitQueueTimeoutMS=2500,
                )
                self.db = self.client[DATABASE_NAME].with_options(codec_options=codec_options)
                logger.info("Connected to MongoDB with connection pooling")
                await self._ensure_schemas()
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

    async def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Closed MongoDB connection")

    async def _ensure_schemas(self):
        """Ensure all required collections exist with proper schemas."""
        for collection_name, schema in SCHEMAS.items():
            try:
                collections = await self.db.list_collection_names()
                if collection_name not in collections:
                    await self.db.create_collection(collection_name, **schema)
                    logger.info(f"Created collection {collection_name} with schema")
                else:
                    await self.db.command("collMod", collection_name, **schema)
                    logger.info(f"Updated schema for collection {collection_name}")
            except Exception as e:
                logger.error(f"Failed to create/update schema for {collection_name}: {e}")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        try:
            yield self.db
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise

    async def insert_prediction(self, prediction_data):
        """Insert a new prediction record."""
        async with self.get_connection() as db:
            try:
                result = await db.predictions.insert_one(prediction_data)
                return result.inserted_id
            except Exception as e:
                logger.error(f"Failed to insert prediction: {e}")
                raise

    async def get_prediction(self, request_id):
        """Get a prediction by request_id."""
        async with self.get_connection() as db:
            try:
                return await db.predictions.find_one({"request_id": request_id})
            except Exception as e:
                logger.error(f"Failed to get prediction: {e}")
                raise

    async def update_rating(self, request_id: str, rating: int) -> bool:
        """Update user rating for a prediction.
        
        Args:
            request_id: The request ID to update
            rating: The user rating (integer)
            
        Returns:
            bool: True if update was successful, False if prediction not found
        """
        async with self.get_connection() as db:
            try:
                result = await db.predictions.update_one(
                    {"request_id": request_id},
                    {"$set": {"user_rating": rating}}
                )
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Failed to update rating: {e}")
                raise

    async def update_transcription(self, request_id: str, transcription: str) -> bool:
        """Update user transcription for a prediction.
        
        Args:
            request_id: The request ID to update
            transcription: The user's transcription text
            
        Returns:
            bool: True if update was successful, False if prediction not found
        """
        async with self.get_connection() as db:
            try:
                result = await db.predictions.update_one(
                    {"request_id": request_id},
                    {"$set": {"user_transcription": transcription}}
                )
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Failed to update transcription: {e}")
                raise

# Create a global database instance
db = Database() 