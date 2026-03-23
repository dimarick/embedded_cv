import ViewportStream from "./ViewportStream.js";
import Viewports from "./Viewports.js";

export default class ViewportInteractiveStream extends ViewportStream {
    #infoImageName;
    #infoImage;
    #valueElement;
    #fragmentSize = {w: 128, h: 128};

    constructor(canvas, socket, valueElement) {
        super(canvas, socket);
        this.#infoImageName = canvas.dataset.infoImage;
        this.#valueElement = valueElement;

        canvas.addEventListener('mousemove', (event) => {
            const x = event.offsetX;
            const y = event.offsetY;

            this.#loadHoverFragment(x, y);
            this.#showHoverValue(x, y);
        })

        canvas.addEventListener('mouseout', (event) => {
            this.#unloadHoverFragment();
        })

        document.addEventListener(Viewports.EVENT_NAME_STREAM_RAW, (event) => this.onRawImage(event))
    }

    onRawImage(event) {
        if (event.detail.header.name === this.#infoImageName) {
            const header = event.detail.header;
            if (header.codec === 0 && header.type === 0x32 && header.channels === 1) {
                this.#infoImage = event.detail;
            }
        }
    }

    renderFrame() {
        super.renderFrame();

        if (!this.#infoImage) {
            return;
        }

        this.renderFloatBuffer(this.element, this.#infoImage.pixelData, 128, 128, 0, 384);
    }

    #loadHoverFragment(x, y) {
        if (!this.imageHeader) {
            return;
        }
        const rect = this.element.getBoundingClientRect();
        const imageX = x / rect.width * this.imageHeader.w - this.#fragmentSize.w / 2;
        const imageY = y / rect.height * this.imageHeader.h - this.#fragmentSize.h / 2;

        this.socket.sendMessage("CHANNEL " + this.quote(this.#infoImageName) +
            " 0 " + this.#fragmentSize.w + " " + this.#fragmentSize.h +
            " " + Math.round(Math.min(this.imageHeader.w - this.#fragmentSize.w, Math.max(0, imageX))) +
            " " + Math.round(Math.min(this.imageHeader.h - this.#fragmentSize.h, Math.max(0, imageY))) + " " +
            this.#fragmentSize.w + " " + this.#fragmentSize.h
        );
    }

    #showHoverValue(x, y) {
        if (!this.#infoImage) {
            return;
        }
        const rect = this.element.getBoundingClientRect();
        const patchOffsetX = this.#infoImage.header.x;
        const patchOffsetY = this.#infoImage.header.y;
        const view = new DataView(this.#infoImage.pixelData);
        const imageX = x / rect.width * this.imageHeader.w;
        const imageY = y / rect.height * this.imageHeader.h;
        const patchX = imageX - patchOffsetX;
        const patchY = imageY - patchOffsetY;

        // поддерживается только RAW CV_32FC1
        if (patchY < 0 || patchY > this.#fragmentSize.h - 4 || patchX < 0 || patchX > this.#fragmentSize.w - 4) {
            this.#valueElement.innerHTML = '';
            return;
        }

        const value = view.getFloat32(Math.round((patchY * this.#fragmentSize.w + patchX)) * 4, true);
        this.#valueElement.innerHTML = value;
    }

    #unloadHoverFragment(x, y) {
        this.#valueElement.innerHTML = '';
        this.socket.sendMessage("DESTROY_CHANNEL " + super.quote(this.#infoImageName) + " 0");
    }

    /**
     * Рендерит одноканальный float-буфер на canvas как grayscale изображение.
     * @param {HTMLCanvasElement} canvas - элемент canvas
     * @param {Float32Array} buffer - данные размером width * height
     * @param {number} width - ширина изображения в пикселях
     * @param {number} height - высота изображения в пикселях
     * @param {number} [minVal] - минимальное значение для нормализации (если не указано, вычисляется из данных)
     * @param {number} [maxVal] - максимальное значение для нормализации (если не указано, вычисляется из данных)
     */
    renderFloatBuffer(canvas, buffer, width, height, minVal, maxVal) {
        if (!canvas || !buffer) return;
        if (buffer.byteLength !== width * height * 4) {
            console.warn('Buffer size mismatch');
            return;
        }

        if (!this.imageHeader) {
            return;
        }

        const view = new DataView(buffer)

        // Вычисляем min/max, если не переданы
        let min = minVal;
        let max = maxVal;
        // if (min === undefined || max === undefined) {
        //     min = Infinity;
        //     max = -Infinity;
        //     for (let i = 0; i < buffer.length; i++) {
        //         const v = buffer[i];
        //         if (v < min) min = v;
        //         if (v > max) max = v;
        //     }
        //     // Защита от случая, когда все значения одинаковы
        //     if (min === max) {
        //         max = min + 1e-6;
        //     }
        // }

        // Создаём ImageData
        const imageData = this.ctx.createImageData(width, height);
        const data = imageData.data; // Uint8ClampedArray, RGBA

        // Заполняем
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const value = view.getFloat32(idx * 4, true);
                // Нормализация в 0..255
                let gray = (value - min) / (max - min);
                gray = Math.min(255, Math.max(0, Math.round(gray * 255)));
                const pixelIdx = idx * 4;
                data[pixelIdx] = gray;     // R
                data[pixelIdx + 1] = gray; // G
                data[pixelIdx + 2] = gray; // B
                data[pixelIdx + 3] = 255;  // A
            }
        }

        this.ctx.putImageData(
            imageData,
            this.#infoImage.header.x / this.imageHeader.w * canvas.width,
            this.#infoImage.header.y / this.imageHeader.h * canvas.height
        );
    }
}