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
        if (patchY < 0 || patchY > this.#infoImage.header.h - 4 || patchX < 0 || patchX > this.#infoImage.header.w - 4) {
            this.#valueElement.innerHTML = '';
            return;
        }

        let pixels = [];
        for (let i = -2; i <= 2; i++) {
            for (let j = -2; j <= 2; j++) {
                const pixel = view.getFloat32(Math.round(((patchY + j) * this.#infoImage.header.w + patchX + i)) * 4, true)
                if (pixel < 0.01) {
                    continue;
                }
                pixels.push(pixel)
            }
        }

        pixels.sort();

        const value = pixels[Math.floor(pixels.length / 2)];
        this.#valueElement.innerHTML = value || '';
    }

    #unloadHoverFragment(x, y) {
        this.#valueElement.innerHTML = '';
        this.socket.sendMessage("DESTROY_CHANNEL " + super.quote(this.#infoImageName) + " 0");
    }
}