import type { ToolOutput } from "./types.ts";

export const MAX_IMAGE_EDGE = 2000;
const MAX_IMAGE_PIXELS = 40_000_000;
// 本工具对单张图片的保守传输预算（Base64 字符数），不是所有模型统一的 API 上限
const MAX_IMAGE_BASE64_CHARS = 4_500_000;

// 按文件头魔数识别 read 支持的图片格式（需至少 12 字节才能识别 WebP）
export function isSupportedImage(head: Buffer): boolean {
  return (
    head.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) ||
    (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) ||
    ["GIF87a", "GIF89a"].includes(head.toString("ascii", 0, 6)) ||
    (head.toString("ascii", 0, 4) === "RIFF" && head.toString("ascii", 8, 12) === "WEBP")
  );
}

// sharp 是原生依赖，按需加载：只用文本工具的宿主不会因平台二进制缺失而无法加载本模块
async function loadSharp(): Promise<typeof import("sharp").default> {
  try {
    return (await import("sharp")).default;
  } catch (err) {
    throw new Error("Image support requires the 'sharp' package, which failed to load.", { cause: err });
  }
}

function base64Length(byteLength: number): number {
  return 4 * Math.ceil(byteLength / 3);
}

// PNG 是否确定为静态图：按 chunk 边界扫描到 IDAT 为止，出现 acTL（APNG 动画声明，规范要求位于 IDAT 之前）
// 或结构无法可靠解析时返回 false。仅用于决定能否原样返回字节，误判为“非静态”只会退回处理分支。
function isStaticPng(buf: Buffer): boolean {
  let offset = 8;
  while (offset + 8 <= buf.length) {
    const length = buf.readUInt32BE(offset);
    const type = buf.toString("latin1", offset + 4, offset + 8);
    if (type === "acTL") return false;
    if (type === "IDAT") return true;
    offset += 12 + length;
  }
  return false;
}

// 自动转正、等比缩放到 MAX_IMAGE_EDGE 以内（不裁剪、不放大），只取第一帧。
// - 已合规的普通静态 JPEG/PNG/WebP（无需转正/缩放、8 位 sRGB 或灰度、无 ICC、体积在预算内）经完整解码校验后
//   原样返回压缩字节，保留原始编码质量；注意这也会保留 EXIF/XMP 等元数据（处理分支会按 sharp 默认移除）。
// - 其余情况输出尽量沿用 API 支持的源格式（JPEG/WebP 原格式，PNG/GIF 输出 PNG）；
//   超出预算时从原图管线重新铺白底转 JPEG，避免在第一次有损结果上再次有损编码。
export async function readImage(buf: Buffer, filePath: string, signal?: AbortSignal): Promise<ToolOutput> {
  const sharp = await loadSharp();
  signal?.throwIfAborted();
  const image = sharp(buf, { pages: 1, page: 0, limitInputPixels: MAX_IMAGE_PIXELS, failOn: "warning" });
  const metadata = await image.metadata();
  signal?.throwIfAborted();

  const format = metadata.format;
  const canPassThrough =
    (format === "jpeg" || format === "png" || format === "webp") &&
    metadata.width > 0 &&
    metadata.height > 0 &&
    metadata.width <= MAX_IMAGE_EDGE &&
    metadata.height <= MAX_IMAGE_EDGE &&
    base64Length(buf.length) <= MAX_IMAGE_BASE64_CHARS &&
    (metadata.orientation ?? 1) === 1 &&
    (metadata.pages ?? 1) === 1 &&
    metadata.depth === "uchar" &&
    (metadata.space === "srgb" || metadata.space === "b-w") &&
    !metadata.hasProfile &&
    (format !== "png" || isStaticPng(buf));
  if (canPassThrough) {
    // metadata() 只读头部；完整解码一次以拒绝截断/损坏的数据，解码出的像素直接丢弃
    await image.clone().raw().toBuffer();
    signal?.throwIfAborted();
    const mediaType = `image/${format}` as const;
    return [
      {
        type: "text",
        text: `Read image ${filePath}: ${metadata.width}x${metadata.height}; sent as ${mediaType} with the original encoding preserved.`,
      },
      { type: "image", source: { type: "base64", media_type: mediaType, data: buf.toString("base64") } },
    ];
  }

  // 基础管线只描述转正与缩放；编码设置只加在 clone() 出的副本上，回退可从同一原图管线重新分支
  const base = image.autoOrient().resize({
    width: MAX_IMAGE_EDGE,
    height: MAX_IMAGE_EDGE,
    fit: "inside",
    withoutEnlargement: true,
  });
  const preferred = base.clone();
  const encoded =
    format === "jpeg"
      ? preferred.jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
      : format === "webp"
        ? preferred.webp({ quality: 90 })
        : preferred.png();
  let result = await encoded.toBuffer({ resolveWithObject: true });
  signal?.throwIfAborted();

  let compressed = false;
  let flattened = false;
  if (base64Length(result.data.length) > MAX_IMAGE_BASE64_CHARS) {
    result = await base
      .clone()
      .flatten({ background: "#ffffff" })
      .jpeg({ quality: 80, chromaSubsampling: "4:2:0" })
      .toBuffer({ resolveWithObject: true });
    signal?.throwIfAborted();
    compressed = true;
    // channels === 4 也可能是 CMYK，透明度以 hasAlpha 为准
    flattened = metadata.hasAlpha === true;
  }
  if (base64Length(result.data.length) > MAX_IMAGE_BASE64_CHARS) {
    throw new Error(
      `Image exceeds ${MAX_IMAGE_BASE64_CHARS} Base64 characters after processing: ${filePath}. Resize it before reading.`
    );
  }
  const data = result.data.toString("base64");

  const mediaType =
    result.info.format === "jpeg" ? "image/jpeg" : result.info.format === "webp" ? "image/webp" : "image/png";
  const notes = [
    `Read image ${filePath}: ${metadata.autoOrient.width}x${metadata.autoOrient.height} after orientation; ` +
      `sent ${result.info.width}x${result.info.height} as ${mediaType}.`,
  ];
  if ((metadata.pages ?? 1) > 1) notes.push("Only the first frame is included.");
  if (compressed) notes.push("JPEG compression was applied to fit the image size budget.");
  if (flattened) notes.push("Transparency was composited onto a white background.");
  return [
    { type: "text", text: notes.join(" ") },
    { type: "image", source: { type: "base64", media_type: mediaType, data } },
  ];
}
