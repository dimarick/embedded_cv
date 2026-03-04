import Viewports from "./Viewports.js";

export default class ViewportStream {
    #ctx;
    #element;
    #name;
    #socket = null;
    #imageData = null;
    #w = 0;
    #h = 0;
    #image = null;

    constructor(canvas, socket) {
        this.#element = canvas;
        this.#name = canvas.dataset.name;
        this.#socket = socket;
        this.#ctx = canvas.getContext("2d");
        this.resize();
        setInterval(() => {
            this.resize();
        }, 1000);

        document.addEventListener(Viewports.EVENT_NAME_STREAM_IMAGE, (event) => this.onImage(event))

        requestAnimationFrame(() => this.renderFrame());
    }

    resize() {
        const r = window.devicePixelRatio;
        const w = this.#element.clientWidth;
        const h = this.#element.clientHeight;

        if ((w !== this.#w && w === 0) || (h !== this.#h && h === 0)) {
            this.#socket.sendMessage("DESTROY_CHANNEL " + this.#quote(this.#name) + " 0");

            this.#w = w;
            this.#h = h;
            return;
        }

        if (Math.abs(w - this.#w) / w < 0.3 && Math.abs(w - this.#w) / w < 0.3) {
            return;
        }

        this.#socket.sendMessage("CHANNEL " + this.#quote(this.#name) + " 0 " + w + " " + h + " 0 0 0 0");

        this.#w = w;
        this.#h = h;
    }

    onImage(event) {
        if (event.detail.header.name !== this.#name) {
            return;
        }
        this.#image = event.detail.image;
    }

    renderFrame() {
        if (this.#image === null) {
            requestAnimationFrame(() => this.renderFrame());
            return;
        }

        const ctx = this.#ctx;

        const r = window.devicePixelRatio;
        const w = this.#element.clientWidth;
        const h = this.#element.clientHeight;

        const iw = this.#image.width;
        const ih = this.#image.height;
        ctx.drawImage(this.#image, (w - iw) / 2 / r, (h - ih) / 2 / r, iw / r, ih / r);

        requestAnimationFrame(() => this.renderFrame());
    }

    #quote(name) {
        return '"' + name.replace(/"/, '\"') + '"';
    }
}