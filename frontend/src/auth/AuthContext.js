import { createContext } from "react";

/** Populated by AuthProvider; read through the useAuth hook. */
export const AuthContext = createContext(null);
