import { useRef } from "react";

import { highlightPython } from "../code/highlight";

/**
 * A live-editing textarea with Python syntax highlighting.
 *
 * The real <textarea> is kept -- typing, selection, the caret, paste, undo,
 * screen readers, everything -- and made transparent. A highlighted copy of
 * the same text sits behind it in a <pre>, which is what the student actually
 * sees; the invisible textarea on top is only there to be typed into. The two
 * layers are the same size (identical font, padding and border, stacked with
 * position: absolute) and their scroll position is copied across on every
 * scroll, so they never drift apart.
 *
 * resize is disabled: a resized textarea with no matching resize on the <pre>
 * behind it would instantly fall out of sync, and the fixed row count already
 * gives ample room.
 *
 * Every prop other than value/onChange/className passes straight through to
 * the underlying textarea, so this is a drop-in replacement for a plain
 * <textarea> wherever the content is Python.
 */
export default function CodeEditor({ value, onChange, className, ...rest }) {
  const textareaRef = useRef(null);
  const preRef = useRef(null);

  // A trailing newline so the last line has somewhere to put its caret when
  // the text ends in "\n" -- otherwise the highlighted layer's last visual
  // line is one short of the textarea's.
  const html = highlightPython(value) + "\n";

  function syncScroll() {
    if (preRef.current && textareaRef.current) {
      preRef.current.scrollTop = textareaRef.current.scrollTop;
      preRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }

  return (
    <div className={`code-editor ${className ?? ""}`.trim()}>
      <pre ref={preRef} className="code-editor__highlight" aria-hidden="true">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
      <textarea
        ref={textareaRef}
        className="code-editor__textarea"
        value={value}
        onChange={onChange}
        onScroll={syncScroll}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        {...rest}
      />
    </div>
  );
}
