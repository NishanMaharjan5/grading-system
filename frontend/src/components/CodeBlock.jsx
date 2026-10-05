import { highlightPython } from "../code/highlight";

/**
 * Read-only Python code, syntax highlighted, with line numbers.
 *
 * Deliberately not line-wrapped -- matches the .code block it replaces,
 * which scrolls sideways rather than wrapping. That keeps one source line
 * equal to one rendered line always, so the gutter can just count 1..N
 * instead of re-measuring where soft-wraps landed.
 */
export default function CodeBlock({ code, className = "" }) {
  const text = code ?? "";
  const lineCount = text === "" ? 1 : text.split("\n").length;
  const html = highlightPython(text);

  return (
    <div className={`code-block ${className}`.trim()}>
      <div className="code-block__gutter" aria-hidden="true">
        {Array.from({ length: lineCount }, (_, i) => (
          <span key={i}>{i + 1}</span>
        ))}
      </div>
      <pre className="code-block__code">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}
