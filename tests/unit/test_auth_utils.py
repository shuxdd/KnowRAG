"""Unit tests for auth utilities — password hashing and JWT creation/verification."""
import pytest


class TestPasswordHashing:
    def test_hash_and_verify(self):
        from backend.utils.auth import hash_password, verify_password

        plain = "mysecret123"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed) is True

    def test_verify_wrong_password(self):
        from backend.utils.auth import hash_password, verify_password

        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False


class TestJWT:
    def test_create_and_decode_token(self):
        from backend.utils.auth import create_access_token, SECRET_KEY, ALGORITHM
        from jose import jwt

        token = create_access_token(data={"sub": "alice"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "alice"
        assert "exp" in payload

    def test_token_expiry(self):
        from backend.utils.auth import create_access_token, SECRET_KEY, ALGORITHM
        from jose import jwt, JWTError
        from datetime import timedelta

        token = create_access_token(
            data={"sub": "alice"}, expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(JWTError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
