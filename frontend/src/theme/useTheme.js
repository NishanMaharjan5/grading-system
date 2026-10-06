import { useSyncExternalStore } from "react";

import { getTheme, setTheme, subscribe } from "./theme";

/**
 * The current theme and a setter. The theme lives outside React (it also
 * changes when the OS does, with no component involved), so it is read
 * through useSyncExternalStore rather than mirrored into state.
 */
export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getTheme, getTheme);
  return [theme, setTheme];
}
