-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_UserAccessGroupMembership" (
    "user_id" TEXT NOT NULL,
    "access_group_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserAccessGroupMembership_pkey" PRIMARY KEY ("user_id","access_group_id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_UserAccessGroupMembership_access_group_id_idx" ON "LiteLLM_UserAccessGroupMembership"("access_group_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_UserAccessGroupMembership" DROP CONSTRAINT IF EXISTS "LiteLLM_UserAccessGroupMembership_user_id_fkey";
ALTER TABLE "LiteLLM_UserAccessGroupMembership" ADD CONSTRAINT "LiteLLM_UserAccessGroupMembership_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserAccessGroupMembership" DROP CONSTRAINT IF EXISTS "LiteLLM_UserAccessGroupMembership_access_group_id_fkey";
ALTER TABLE "LiteLLM_UserAccessGroupMembership" ADD CONSTRAINT "LiteLLM_UserAccessGroupMembership_access_group_id_fkey" FOREIGN KEY ("access_group_id") REFERENCES "LiteLLM_AccessGroupTable"("access_group_id") ON DELETE RESTRICT ON UPDATE CASCADE;
