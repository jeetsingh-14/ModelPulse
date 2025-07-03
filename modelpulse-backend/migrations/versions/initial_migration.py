"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2025-07-03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create inference_logs table
    op.create_table('inference_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('input_shape', sa.String(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('output_class', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_inference_logs_id'), 'inference_logs', ['id'], unique=False)
    op.create_index(op.f('ix_inference_logs_model_name'), 'inference_logs', ['model_name'], unique=False)
    op.create_index(op.f('ix_inference_logs_output_class'), 'inference_logs', ['output_class'], unique=False)


def downgrade():
    # Drop indexes
    op.drop_index(op.f('ix_inference_logs_output_class'), table_name='inference_logs')
    op.drop_index(op.f('ix_inference_logs_model_name'), table_name='inference_logs')
    op.drop_index(op.f('ix_inference_logs_id'), table_name='inference_logs')
    
    # Drop table
    op.drop_table('inference_logs')