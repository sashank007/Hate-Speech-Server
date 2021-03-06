"""new body

Revision ID: 47e2bd6cd010
Revises: f4f1b0566c47
Create Date: 2019-05-03 09:47:01.776000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '47e2bd6cd010'
down_revision = 'f4f1b0566c47'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('post', sa.Column('body', sa.String(length=140), nullable=True))
    op.drop_index('ix_post_post', table_name='post')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index('ix_post_post', 'post', ['post'], unique=False)
    op.drop_column('post', 'body')
    # ### end Alembic commands ###
