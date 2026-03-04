import Socket from "./Socket";
import Viewports from "./Viewports";

export default class ViewportStream {
    #ctx;
    #element;
    #name;
    #socket = null;
    #imageData = null;
    #w = 0;
    #h = 0;
    #image = null;

    constructor(element, socket) {
        this.#element = element;
        this.#name = element.dataset.name;
        this.#socket = socket;
        this.#ctx = canvas.getContext("2d");
        this.resize();
        setInterval(() => {
            this.resize();
        }, 1000);

        document.addEventListener(Viewports.EVENT_NAME_STREAM_IMAGE, onImage)

        requestAnimationFrame(this.renderFrame);
    }

    resize() {
        const w = this.#element.width;
        const h = this.#element.height;

        if ((w !== this.#w && w === 0) || (h !== this.#h && h === 0)) {
            this.#socket.sendMessage("DESTROY_CHANNEL debug__0 0");

            this.#w = w;
            this.#h = h;
            return;
        }

        if (Math.abs(w - this.#w) / w < 0.3 && Math.abs(w - this.#w) / w < 0.3) {
            return;
        }

        if (this.#socket !== null) {
            this.#socket.close();
        }

        this.#socket.sendMessage("CHANNEL debug__0 0 0 " + w + " " + h + " 1");

        this.#w = w;
        this.#h = h;
    }

    onImage(event) {
        if (event.detail.name === this.name) {
            return;
        }
        this.#image = event.detail.image;
    }

    renderFrame() {
        if (this.#image === null) {
            requestAnimationFrame(this.renderFrame);
        }

        const ctx = this.#ctx;

        ctx.drawImage(this.#image);

        requestAnimationFrame(this.renderFrame);
    }
}