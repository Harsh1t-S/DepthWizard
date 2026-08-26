import type { DepthGrid } from "../types/api";

/**
 * A height field decoded from the backend's 16-bit PNG.
 *
 * The mesh grid decides how sharp the reconstruction can look. A JSON float
 * array is too heavy to ship at useful resolution, so the sharp grid arrives as
 * a quantised image and is decoded here into plain floats.
 */
export interface DecodedHeightGrid {
  width: number;
  height: number;
  /** Row-major heights in the source units. */
  values: Float32Array;
  /** Row-major validity, or null when every cell is valid. */
  valid: Uint8Array | null;
}

function drawToContext(image: HTMLImageElement): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D context unavailable");
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load height image: ${url}`));
    image.src = url;
  });
}

/**
 * Decode the high-resolution grid, or return null to fall back to `values`.
 *
 * Browsers decode a 16-bit greyscale PNG down to 8 bits per channel when it is
 * drawn to a canvas, and the high byte lands in every channel. Reading red
 * therefore recovers the top 8 bits, which is a step of about 1/255 of the
 * scene range -- roughly 25 cm over a 65 m spread, well below the model's own
 * error and far finer than the resolution gained.
 */
export async function decodeHeightGrid(
  grid: DepthGrid,
  resolveUrl: (value: string | null | undefined) => string | null,
): Promise<DecodedHeightGrid | null> {
  const encoded = grid.encoded;
  if (!encoded || encoded.format !== "png16") return null;

  const url = resolveUrl(encoded.url);
  if (!url) return null;

  const image = await loadImage(url);
  const pixels = drawToContext(image);
  const { width, height } = pixels;
  const span = encoded.maximum - encoded.minimum;
  const values = new Float32Array(width * height);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = encoded.minimum + (pixels.data[i * 4] / 255) * span;
  }

  let valid: Uint8Array | null = null;
  const maskUrl = resolveUrl(encoded.mask_url);
  if (maskUrl) {
    try {
      const maskPixels = drawToContext(await loadImage(maskUrl));
      valid = new Uint8Array(width * height);
      for (let i = 0; i < valid.length; i += 1) {
        valid[i] = maskPixels.data[i * 4] > 127 ? 1 : 0;
      }
    } catch {
      // A missing mask is not fatal; treat every cell as valid.
      valid = null;
    }
  }

  return { width, height, values, valid };
}
