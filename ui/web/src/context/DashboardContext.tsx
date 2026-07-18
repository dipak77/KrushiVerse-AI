import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "../lib/api";

type Bootstrap = any;

const Ctx = createContext<{
  bootstrap: Bootstrap | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}>({ bootstrap: null, loading: true, error: null, refresh: () => {} });

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const b = await api.bootstrap("FARM_101");
      setBootstrap(b);
      setError(null);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return <Ctx.Provider value={{ bootstrap, loading, error, refresh: load }}>{children}</Ctx.Provider>;
}

export function useDashboard() {
  return useContext(Ctx);
}
