"""
S3/MinIO storage integration for raw HTML archival
"""
import boto3
from botocore.exceptions import ClientError
from typing import Optional, List
import logging
from datetime import datetime
import gzip
import json

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================
# S3 CLIENT INITIALIZATION
# ============================================

class S3Storage:
    """S3/MinIO storage manager"""

    def __init__(self):
        self.client = None
        self.bucket_name = settings.S3_BUCKET_NAME
        self._initialize_client()

    def _initialize_client(self):
        """Initialize S3 client"""
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION,
                use_ssl=settings.S3_USE_SSL
            )
            logger.info(f"S3 client initialized for endpoint: {settings.S3_ENDPOINT_URL}")

            # Create bucket if it doesn't exist
            self._ensure_bucket_exists()
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")
            raise

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    self.client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' created")
                except Exception as create_error:
                    logger.error(f"Error creating bucket: {create_error}")
            else:
                logger.error(f"Error checking bucket: {e}")

    def health_check(self) -> bool:
        """Check if S3 is accessible"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return False


# Global storage instance
s3_storage = S3Storage()


# ============================================
# HTML STORAGE
# ============================================

async def store_html(
    url: str,
    html: str,
    page_id: int,
    compress: bool = True,
    metadata: dict = None
) -> Optional[str]:
    """
    Store raw HTML in S3

    Args:
        url: The page URL
        html: Raw HTML content
        page_id: Database page ID
        compress: Gzip compress before storing
        metadata: Additional metadata

    Returns:
        S3 key (path) if successful
    """
    try:
        # Generate S3 key with date-based path
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        s3_key = f"html/{date_path}/page_{page_id}.html"

        # Compress HTML if requested
        content = html.encode('utf-8')
        if compress:
            content = gzip.compress(content)
            s3_key += ".gz"

        # Prepare metadata
        s3_metadata = {
            "url": url[:1000],  # S3 metadata has size limits
            "page_id": str(page_id),
            "scraped_at": datetime.utcnow().isoformat(),
            "compressed": str(compress).lower()
        }

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    s3_metadata[key] = str(value)

        # Upload to S3
        s3_storage.client.put_object(
            Bucket=s3_storage.bucket_name,
            Key=s3_key,
            Body=content,
            Metadata=s3_metadata,
            ContentType='text/html' if not compress else 'application/gzip'
        )

        logger.info(f"HTML stored in S3: {s3_key}")
        return s3_key

    except Exception as e:
        logger.error(f"Error storing HTML in S3: {e}")
        return None


async def retrieve_html(s3_key: str, decompress: bool = None) -> Optional[str]:
    """
    Retrieve HTML from S3

    Args:
        s3_key: S3 object key
        decompress: Auto-detect from key if None

    Returns:
        HTML content
    """
    try:
        # Get object from S3
        response = s3_storage.client.get_object(
            Bucket=s3_storage.bucket_name,
            Key=s3_key
        )

        content = response['Body'].read()

        # Decompress if needed
        if decompress is None:
            decompress = s3_key.endswith('.gz')

        if decompress:
            content = gzip.decompress(content)

        return content.decode('utf-8')

    except Exception as e:
        logger.error(f"Error retrieving HTML from S3: {e}")
        return None


async def delete_html(s3_key: str) -> bool:
    """Delete HTML from S3"""
    try:
        s3_storage.client.delete_object(
            Bucket=s3_storage.bucket_name,
            Key=s3_key
        )
        logger.info(f"HTML deleted from S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error deleting HTML from S3: {e}")
        return False


# ============================================
# METADATA STORAGE
# ============================================

async def store_metadata(
    page_id: int,
    metadata: dict
) -> Optional[str]:
    """Store page metadata as JSON in S3"""
    try:
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        s3_key = f"metadata/{date_path}/page_{page_id}.json"

        json_content = json.dumps(metadata, indent=2)

        s3_storage.client.put_object(
            Bucket=s3_storage.bucket_name,
            Key=s3_key,
            Body=json_content.encode('utf-8'),
            ContentType='application/json'
        )

        logger.debug(f"Metadata stored in S3: {s3_key}")
        return s3_key

    except Exception as e:
        logger.error(f"Error storing metadata in S3: {e}")
        return None


# ============================================
# BATCH OPERATIONS
# ============================================

async def store_html_batch(pages: List[dict]) -> List[dict]:
    """
    Store multiple HTML pages in batch

    Args:
        pages: List of dicts with keys: url, html, page_id, metadata

    Returns:
        List of results with s3_key added
    """
    results = []
    for page in pages:
        s3_key = await store_html(
            url=page['url'],
            html=page['html'],
            page_id=page['page_id'],
            metadata=page.get('metadata')
        )
        results.append({
            **page,
            's3_key': s3_key,
            'success': s3_key is not None
        })
    return results


# ============================================
# STORAGE STATISTICS
# ============================================

async def get_storage_stats() -> dict:
    """Get storage statistics"""
    try:
        # List objects in bucket
        response = s3_storage.client.list_objects_v2(
            Bucket=s3_storage.bucket_name
        )

        total_size = 0
        object_count = 0

        if 'Contents' in response:
            for obj in response['Contents']:
                total_size += obj['Size']
                object_count += 1

        return {
            "bucket": s3_storage.bucket_name,
            "object_count": object_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return {}


async def cleanup_old_objects(days: int = 30) -> int:
    """
    Delete objects older than specified days

    Returns:
        Number of objects deleted
    """
    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        response = s3_storage.client.list_objects_v2(
            Bucket=s3_storage.bucket_name
        )

        deleted_count = 0

        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    s3_storage.client.delete_object(
                        Bucket=s3_storage.bucket_name,
                        Key=obj['Key']
                    )
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old objects")
        return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up old objects: {e}")
        return 0
