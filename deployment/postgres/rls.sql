-- Enable Row-Level Security for multi-tenant SaaS
-- Usage:
--   psql -d constraintlattice -f deployment/postgres/rls.sql

ALTER TABLE public.audit_trace ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON public.audit_trace
USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Helper function to set tenant in session
CREATE OR REPLACE FUNCTION set_tenant(tenant uuid) RETURNS void AS $$
BEGIN
  PERFORM set_config('app.current_tenant', tenant::text, true);
END
$$ LANGUAGE plpgsql;
