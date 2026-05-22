"""
认证路由模块

提供用户注册、登录、获取当前用户信息等认证相关接口。

接口列表：
- POST /api/auth/register: 用户注册
- POST /api/auth/login: 用户登录，获取 JWT 令牌
- GET /api/auth/me: 获取当前登录用户信息

安全措施：
- 密码使用 bcrypt 哈希存储
- 登录时使用虚拟哈希防止用户枚举攻击
- JWT 令牌有过期时间
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from backend.db import SessionFactory
from backend.models.db_models import UserORM
from backend.models.schemas import (
    AuthRegisterRequest,
    AuthLoginRequest,
    AuthUserResponse,
    AuthTokenResponse,
)
from backend.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

DUMMY_HASH = "$2b$12$pF.Qp0KPLYyhnClcRfKLiO97K2CHD1Qj5wWVk5tS9EV72Ua1p6Ftq"


@router.post("/register", response_model=AuthUserResponse)
async def register(req: AuthRegisterRequest):
    """
    用户注册接口

    Args:
        req: 包含用户名和密码的注册请求

    Returns:
        新创建的用户信息

    Raises:
        HTTPException 409: 用户名已存在
    """
    with SessionFactory() as session:
        user = UserORM(
            username=req.username,
            hashed_password=hash_password(req.password),
        )
        session.add(user)
        try:
            session.commit()
            session.refresh(user)
        except IntegrityError:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        return AuthUserResponse(
            id=user.id,
            username=user.username,
            created_at=user.created_at.isoformat() if user.created_at else "",
        )


@router.post("/login", response_model=AuthTokenResponse)
async def login(req: AuthLoginRequest):
    """
    用户登录接口

    Args:
        req: 包含用户名和密码的登录请求

    Returns:
        JWT 访问令牌

    Raises:
        HTTPException 401: 用户名或密码错误
    """
    with SessionFactory() as session:
        user = (
            session.query(UserORM)
            .filter(UserORM.username == req.username)
            .first()
        )
        if user is None:
            verify_password(req.password, DUMMY_HASH)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        if not verify_password(req.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        token = create_access_token(data={"sub": user.username})
        return AuthTokenResponse(access_token=token)


@router.get("/me", response_model=AuthUserResponse)
async def me(current_user=Depends(get_current_user)):
    """
    获取当前登录用户信息接口

    Args:
        current_user: 通过 JWT 令牌自动注入的当前用户

    Returns:
        当前用户的信息
    """
    return AuthUserResponse(
        id=current_user.id,
        username=current_user.username,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
    )
