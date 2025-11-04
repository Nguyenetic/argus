"""
Hybrid search implementation using Reciprocal Rank Fusion (RRF)
Combines vector similarity search + full-text search for 30-50% better accuracy
Based on SurfSense and sage-mcp implementations
"""
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
import logging

from storage.models import ScrapedPage, PageChunk
from storage.embeddings import get_embedding

logger = logging.getLogger(__name__)


# ============================================
# RRF CONSTANTS
# ============================================

# RRF constant from research (R2R paper)
RRF_K = 60

# Default weights for hybrid search
DEFAULT_SEMANTIC_WEIGHT = 5.0  # Prioritize semantic results
DEFAULT_KEYWORD_WEIGHT = 1.0   # Keyword supplements


# ============================================
# HYBRID SEARCH (RRF Algorithm)
# ============================================

async def hybrid_search_pages(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid search on scraped pages using RRF algorithm

    Args:
        session: Database session
        query: Search query
        top_k: Number of results to return
        semantic_weight: Weight for semantic/vector search
        keyword_weight: Weight for keyword/full-text search
        filters: Optional filters (status, tags, etc.)

    Returns:
        List of pages with RRF scores

    Algorithm:
        1. Generate query embedding
        2. Vector search (top_k * 2 results)
        3. Full-text search (top_k * 2 results)
        4. RRF fusion: score = semantic_weight/(k+rank) + keyword_weight/(k+rank)
        5. Sort by combined score and return top_k
    """
    try:
        # Step 1: Generate query embedding
        query_embedding = await get_embedding(query)

        # Step 2: Build semantic search CTE
        semantic_cte = (
            select(
                ScrapedPage.id,
                func.rank().over(
                    order_by=ScrapedPage.embedding.op("<=>")(query_embedding)
                ).label("rank")
            )
            .where(ScrapedPage.status == "completed")
        )

        # Apply filters to semantic search
        if filters:
            semantic_cte = _apply_filters(semantic_cte, ScrapedPage, filters)

        semantic_cte = semantic_cte.limit(top_k * 2).cte("semantic_search")

        # Step 3: Build keyword search CTE
        tsvector = func.to_tsvector("english", ScrapedPage.content)
        tsquery = func.plainto_tsquery("english", query)

        keyword_cte = (
            select(
                ScrapedPage.id,
                func.rank().over(
                    order_by=func.ts_rank_cd(tsvector, tsquery).desc()
                ).label("rank")
            )
            .where(tsvector.op("@@")(tsquery))
            .where(ScrapedPage.status == "completed")
        )

        # Apply filters to keyword search
        if filters:
            keyword_cte = _apply_filters(keyword_cte, ScrapedPage, filters)

        keyword_cte = keyword_cte.limit(top_k * 2).cte("keyword_search")

        # Step 4: RRF Fusion
        # score = semantic_weight/(RRF_K + semantic_rank) + keyword_weight/(RRF_K + keyword_rank)
        final_query = (
            select(
                ScrapedPage,
                (
                    func.coalesce(
                        semantic_weight / (RRF_K + semantic_cte.c.rank), 0.0
                    ) +
                    func.coalesce(
                        keyword_weight / (RRF_K + keyword_cte.c.rank), 0.0
                    )
                ).label("rrf_score")
            )
            .select_from(
                semantic_cte.outerjoin(
                    keyword_cte,
                    semantic_cte.c.id == keyword_cte.c.id,
                    full=True
                )
            )
            .join(
                ScrapedPage,
                ScrapedPage.id.in_([
                    func.coalesce(semantic_cte.c.id, keyword_cte.c.id)
                ])
            )
            .order_by(text("rrf_score DESC"))
            .limit(top_k)
        )

        # Execute query
        result = await session.execute(final_query)
        rows = result.all()

        # Format results
        results = []
        for row in rows:
            page = row[0]
            score = row[1]
            results.append({
                "id": page.id,
                "url": page.url,
                "title": page.title,
                "content": page.content[:500] + "..." if page.content else None,
                "rrf_score": float(score),
                "metadata": page.metadata,
                "tags": page.tags,
                "scraped_at": page.scraped_at.isoformat() if page.scraped_at else None,
            })

        logger.info(f"Hybrid search returned {len(results)} results for query: {query}")
        return results

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise


async def hybrid_search_chunks(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
) -> List[Dict[str, Any]]:
    """
    Hybrid search on page chunks (more granular than full pages)

    Returns chunks with their parent page information
    """
    try:
        # Generate query embedding
        query_embedding = await get_embedding(query)

        # Semantic search on chunks
        semantic_cte = (
            select(
                PageChunk.id,
                func.rank().over(
                    order_by=PageChunk.embedding.op("<=>")(query_embedding)
                ).label("rank")
            )
            .limit(top_k * 2)
            .cte("semantic_search")
        )

        # Keyword search on chunks
        tsvector = func.to_tsvector("english", PageChunk.content)
        tsquery = func.plainto_tsquery("english", query)

        keyword_cte = (
            select(
                PageChunk.id,
                func.rank().over(
                    order_by=func.ts_rank_cd(tsvector, tsquery).desc()
                ).label("rank")
            )
            .where(tsvector.op("@@")(tsquery))
            .limit(top_k * 2)
            .cte("keyword_search")
        )

        # RRF fusion
        final_query = (
            select(
                PageChunk,
                ScrapedPage,
                (
                    func.coalesce(
                        semantic_weight / (RRF_K + semantic_cte.c.rank), 0.0
                    ) +
                    func.coalesce(
                        keyword_weight / (RRF_K + keyword_cte.c.rank), 0.0
                    )
                ).label("rrf_score")
            )
            .select_from(
                semantic_cte.outerjoin(
                    keyword_cte,
                    semantic_cte.c.id == keyword_cte.c.id,
                    full=True
                )
            )
            .join(
                PageChunk,
                PageChunk.id.in_([
                    func.coalesce(semantic_cte.c.id, keyword_cte.c.id)
                ])
            )
            .join(ScrapedPage, PageChunk.page_id == ScrapedPage.id)
            .order_by(text("rrf_score DESC"))
            .limit(top_k)
        )

        result = await session.execute(final_query)
        rows = result.all()

        # Format results
        results = []
        for row in rows:
            chunk = row[0]
            page = row[1]
            score = row[2]
            results.append({
                "chunk_id": chunk.id,
                "chunk_content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_id": page.id,
                "page_url": page.url,
                "page_title": page.title,
                "rrf_score": float(score),
            })

        logger.info(f"Chunk hybrid search returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Error in chunk hybrid search: {e}")
        raise


# ============================================
# PURE SEMANTIC SEARCH (Fallback)
# ============================================

async def semantic_search_pages(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Pure vector similarity search (fallback when full-text not needed)
    """
    try:
        query_embedding = await get_embedding(query)

        query_stmt = (
            select(ScrapedPage)
            .where(ScrapedPage.status == "completed")
            .order_by(ScrapedPage.embedding.op("<=>")(query_embedding))
            .limit(top_k)
        )

        if filters:
            query_stmt = _apply_filters(query_stmt, ScrapedPage, filters)

        result = await session.execute(query_stmt)
        pages = result.scalars().all()

        return [_format_page_result(page) for page in pages]

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise


# ============================================
# PURE KEYWORD SEARCH (Fallback)
# ============================================

async def keyword_search_pages(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Pure full-text search (fallback for exact keyword matching)
    """
    try:
        tsvector = func.to_tsvector("english", ScrapedPage.content)
        tsquery = func.plainto_tsquery("english", query)

        query_stmt = (
            select(
                ScrapedPage,
                func.ts_rank_cd(tsvector, tsquery).label("rank")
            )
            .where(tsvector.op("@@")(tsquery))
            .where(ScrapedPage.status == "completed")
            .order_by(text("rank DESC"))
            .limit(top_k)
        )

        if filters:
            query_stmt = _apply_filters(query_stmt, ScrapedPage, filters)

        result = await session.execute(query_stmt)
        rows = result.all()

        return [_format_page_result(row[0], {"fts_rank": float(row[1])}) for row in rows]

    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        raise


# ============================================
# HELPER FUNCTIONS
# ============================================

def _apply_filters(query, model, filters: Dict[str, Any]):
    """Apply filters to query"""
    for key, value in filters.items():
        if hasattr(model, key):
            if isinstance(value, list):
                query = query.where(getattr(model, key).in_(value))
            else:
                query = query.where(getattr(model, key) == value)
    return query


def _format_page_result(page: ScrapedPage, extra: Dict = None) -> Dict[str, Any]:
    """Format page result"""
    result = {
        "id": page.id,
        "url": page.url,
        "title": page.title,
        "content": page.content[:500] + "..." if page.content else None,
        "metadata": page.metadata,
        "tags": page.tags,
        "scraped_at": page.scraped_at.isoformat() if page.scraped_at else None,
    }
    if extra:
        result.update(extra)
    return result
