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

# Dummy bcrypt hash used to prevent timing-based username enumeration on login.
# When a user doesn't exist, we still run bcrypt verify to keep response timing constant.
DUMMY_HASH = "$2b$12$pF.Qp0KPLYyhnClcRfKLiO97K2CHD1Qj5wWVk5tS9EV72Ua1p6Ftq"


@router.post("/register", response_model=AuthUserResponse)
async def register(req: AuthRegisterRequest):
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
    with SessionFactory() as session:
        user = (
            session.query(UserORM)
            .filter(UserORM.username == req.username)
            .first()
        )
        if user is None:
            # Do NOT reveal whether username exists — dummy-verify to prevent timing leak
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
    return AuthUserResponse(
        id=current_user.id,
        username=current_user.username,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
    )
