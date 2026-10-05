/**
 * The one place highlight.js is configured, so every code display in the app
 * tokenises the same way.
 *
 * Only the Python language grammar is registered -- this app grades Python
 * submissions and nothing else, so pulling in the rest of highlight.js's
 * language set would be dead weight in the bundle.
 *
 * highlight.js HTML-escapes the source text it wraps in <span> tags (that is
 * how it renders `<`, `>` and `&` safely), so the result of highlightPython()
 * is safe to render with dangerouslySetInnerHTML. It never passes submitted
 * text through unescaped.
 */
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("python", python);

export function highlightPython(code) {
  if (!code) return "";
  return hljs.highlight(code, { language: "python", ignoreIllegals: true }).value;
}
