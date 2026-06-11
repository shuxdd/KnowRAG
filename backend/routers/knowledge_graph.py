"""知识图谱浏览 API 路由"""

import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from backend.models.schemas import (
    KGStatsResponse, KGEntityListResponse, KGEntitySummary,
    KGEntityDetailResponse, KGRelationInfo, KGChunkRef,
)
from backend.utils.auth import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["knowledge-graph"])


@router.get("/stats", response_model=KGStatsResponse)
async def get_kg_stats(current_user: CurrentUser = Depends(get_current_user)):
    from backend.services.graph_service import graph_service
    stats = await asyncio.to_thread(graph_service.get_stats, user_id=current_user.id)
    return KGStatsResponse(**stats)


@router.get("/entities", response_model=KGEntityListResponse)
async def list_entities(
    entity_type: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 20,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    result = await asyncio.to_thread(
        graph_service.get_entities,
        user_id=current_user.id, entity_type=entity_type,
        search=search, page=page, page_size=page_size,
    )
    return KGEntityListResponse(**result)


@router.get("/entities/{entity_id}", response_model=KGEntityDetailResponse)
async def get_entity_detail(
    entity_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    result = await asyncio.to_thread(
        graph_service.get_entity_detail, entity_id=entity_id, user_id=current_user.id,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Entity not found")
    return KGEntityDetailResponse(**result)


@router.get("/search")
async def search_kg(
    q: str = "",
    top_k: int = 10,
    current_user: CurrentUser = Depends(get_current_user),
):
    from backend.services.graph_service import graph_service
    from backend.services.entity_extractor import entity_extractor

    entities = await asyncio.to_thread(entity_extractor.extract_query_entities, q)
    if not entities:
        return {"entities": [], "chunks": []}

    chunk_ids = await asyncio.to_thread(
        graph_service.search_by_entities, entities, user_id=current_user.id, top_k=top_k,
    )
    return {"matched_entities": entities, "chunk_ids": chunk_ids}


@router.post("/extract/{doc_id}")
async def manual_extract(
    doc_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """手动触发文档全量抽取（绕过阈值限制）。"""
    from backend.services.parent_store import parent_store
    from backend.services.graph_service import graph_service
    from backend.services.entity_extractor import entity_extractor

    parents = await asyncio.to_thread(parent_store.get_by_filename, doc_id, user_id=current_user.id)
    if not parents:
        raise HTTPException(status_code=404, detail="Document not found")

    extracted_count = 0
    for p in parents:
        result = await asyncio.to_thread(entity_extractor.extract_from_chunk, p.content)
        if result["entities"]:
            await asyncio.to_thread(
                graph_service.build_graph_for_chunk,
                chunk_id=p.id, filename=p.filename,
                heading_path=json.dumps(p.heading_path, ensure_ascii=False),
                entities=result["entities"], relations=result["relations"],
                user_id=current_user.id,
            )
            extracted_count += 1

    return {"detail": f"Extracted entities from {extracted_count}/{len(parents)} chunks"}
