from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import ValidationError

from .database import get_db
from .models import User, UserRole
from .schemas import TokenData

# JWT Configuration
SECRET_KEY = "YOUR_SECRET_KEY_HERE"  # In production, use a secure environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None, organization_id: Optional[int] = None, org_role: Optional[str] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time
        organization_id: Optional organization ID for tenant context
        org_role: Optional role in the organization

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    # Add organization context if provided
    if organization_id is not None:
        to_encode.update({"organization_id": organization_id})

    # Add organization role if provided
    if org_role is not None:
        to_encode.update({"org_role": org_role})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData object with username, role, organization_id, and org_role

    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        organization_id: Optional[int] = payload.get("organization_id")
        org_role: Optional[str] = payload.get("org_role")

        if username is None:
            raise credentials_exception

        token_data = TokenData(
            username=username, 
            role=role, 
            organization_id=organization_id, 
            org_role=org_role
        )
        return token_data

    except (JWTError, ValidationError):
        raise credentials_exception


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user.

    Args:
        token: JWT token from request
        db: Database session

    Returns:
        User object with organization context

    Raises:
        HTTPException: If user not found or inactive
    """
    token_data = verify_token(token)
    user = db.query(User).filter(User.username == token_data.username).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Add organization context to user object
    user.current_organization_id = token_data.organization_id
    user.current_org_role = token_data.org_role

    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active user.

    Args:
        current_user: User from get_current_user dependency

    Returns:
        User object

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


# Role-based access control dependencies
def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Check if the current user has admin role (global or in current organization).

    Args:
        current_user: User from get_current_user dependency

    Returns:
        User object

    Raises:
        HTTPException: If user doesn't have admin role
    """
    # Check global admin role (for backward compatibility)
    if current_user.role == UserRole.ADMIN:
        return current_user

    # Check organization-specific admin role
    if hasattr(current_user, 'current_organization_id') and current_user.current_organization_id:
        if current_user.current_org_role == UserRole.ADMIN:
            return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
    )


def get_analyst_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Check if the current user has analyst or admin role (global or in current organization).

    Args:
        current_user: User from get_current_user dependency

    Returns:
        User object

    Raises:
        HTTPException: If user doesn't have analyst or admin role
    """
    # Check global role (for backward compatibility)
    if current_user.role in [UserRole.ADMIN, UserRole.ANALYST]:
        return current_user

    # Check organization-specific role
    if hasattr(current_user, 'current_organization_id') and current_user.current_organization_id:
        if current_user.current_org_role in [UserRole.ADMIN, UserRole.ANALYST]:
            return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
    )


def get_organization_member(current_user: User = Depends(get_current_user)) -> User:
    """
    Check if the current user is a member of the current organization.

    Args:
        current_user: User from get_current_user dependency

    Returns:
        User object

    Raises:
        HTTPException: If user is not a member of the current organization
    """
    if not hasattr(current_user, 'current_organization_id') or not current_user.current_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No organization context provided"
        )

    # The user has already been authenticated with an organization context
    # If we got this far, they are a member of the organization
    return current_user
