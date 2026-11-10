export function errnoCode(err: unknown): string | undefined {
  return (err as NodeJS.ErrnoException | null)?.code;
}

// 避免在 JavaScript 字符串截断处切开 UTF-16 代理对（如 Emoji）
export function sliceStringSafe(str: string, maxChars: number): string {
  if (str.length <= maxChars) return str;
  let end = maxChars;
  const code = str.charCodeAt(end - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    end--;
  }
  return str.slice(0, end);
}
