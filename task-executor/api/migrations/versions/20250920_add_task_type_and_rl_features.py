"""Add task_type column and RL training features

Revision ID: 002_rl_features
Revises: 20250914_153147_001_initial_tables
Create Date: 2025-09-20 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_rl_features'
down_revision = '20250914_153147_001_initial_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add task_type column to tasks table
    op.add_column('tasks', sa.Column('task_type', 
                                   sa.Enum('standard', 'rl_training', name='task_type', create_type=True),
                                   nullable=False,
                                   server_default='standard',
                                   comment='Task execution type'))
    
    # Add index for task_type and status combination
    op.create_index('idx_tasks_type_status', 'tasks', ['task_type', 'status'])
    
    # Create rl_interactions table if it doesn't exist
    op.create_table('rl_interactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False, comment='Associated task ID'),
        sa.Column('step_number', sa.Integer(), nullable=False, comment='Sequential step number in the RL episode'),
        sa.Column('shell_command', sa.Text(), nullable=False, comment='Shell command executed by the RL agent'),
        sa.Column('env_response', postgresql.JSONB(astext_type=sa.Text()), nullable=False, default={}, 
                 comment='Environment response including metrics, logs, and system state'),
        sa.Column('judge_score', sa.Float(), nullable=True, comment='Judge evaluation score (0.0-1.0)'),
        sa.Column('judge_feedback', sa.Text(), nullable=True, comment='Detailed feedback from the LLM judge'),
        sa.Column('execution_duration', sa.Float(), nullable=True, comment='Command execution duration in seconds'),
        sa.Column('exit_code', sa.Integer(), nullable=True, comment='Shell command exit code'),
        sa.Column('stdout', sa.Text(), nullable=True, comment='Standard output from command execution'),
        sa.Column('stderr', sa.Text(), nullable=True, comment='Standard error from command execution'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(),
                 comment='When the interaction was recorded'),
        sa.Column('executed_at', sa.DateTime(timezone=True), nullable=True, comment='When the command was executed'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='CASCADE')
    )
    
    # Create indexes for rl_interactions table
    op.create_index('idx_rl_task_step', 'rl_interactions', ['task_id', 'step_number'])
    op.create_index('idx_rl_created_at', 'rl_interactions', ['created_at'])
    op.create_index('idx_rl_judge_score', 'rl_interactions', ['judge_score'])
    op.create_index(op.f('ix_rl_interactions_task_id'), 'rl_interactions', ['task_id'])


def downgrade() -> None:
    # Drop rl_interactions table and indexes
    op.drop_index(op.f('ix_rl_interactions_task_id'), table_name='rl_interactions')
    op.drop_index('idx_rl_judge_score', table_name='rl_interactions')
    op.drop_index('idx_rl_created_at', table_name='rl_interactions')
    op.drop_index('idx_rl_task_step', table_name='rl_interactions')
    op.drop_table('rl_interactions')
    
    # Drop task_type column and related objects
    op.drop_index('idx_tasks_type_status', table_name='tasks')
    op.drop_column('tasks', 'task_type')
    
    # Drop the enum type
    op.execute("DROP TYPE IF EXISTS task_type")