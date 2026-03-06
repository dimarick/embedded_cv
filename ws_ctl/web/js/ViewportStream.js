import Viewports from "./Viewports.js";

export default class ViewportStream {
    #ctx;
    #element;
    #name;
    #socket = null;
    #imageData = null;
    #w = 0;
    #h = 0;
    #visible = false;
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
        const rect = this.#element.getBoundingClientRect();
        const w = rect.width * r
        const h = rect.height * r;

        const visible = this.isVisible();
        if (!visible) {
            if (this.#visible !== visible) {
                this.#socket.sendMessage("DESTROY_CHANNEL " + this.#quote(this.#name) + " 0");
                this.#visible = visible;
            }

            return;
        }
        if (this.#visible === visible && Math.abs(w - this.#w) / w < 0.3 && Math.abs(w - this.#w) / w < 0.3) {
            return;
        }

        this.#visible = visible;

        this.#socket.sendMessage("CHANNEL " + this.#quote(this.#name) + " 0 " + Math.round(w) + " " + Math.round(h) + " 0 0 0 0");

        this.#w = w;
        this.#h = h;

        const dpr = window.devicePixelRatio;
        this.#element.width = Math.round(w * dpr);
        this.#element.height = Math.round(h * dpr);
    }

    isVisible() {
        return !document.hidden && this.#isElementVisibleByCSS(this.#element) && this.#isElementInViewPort(this.#element);
    }
    #isElementVisibleByCSS(element) {
        const style = getComputedStyle(element)
        const rect = element.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0 || style.visibility === 'hidden' || style.opacity === '0' || element.hidden) {
            return false;
        }

        // Проверка родителей (visibility может наследоваться)
        const parent = element.parentElement;

        if (!parent) {
            return true;
        }

        return this.#isElementVisibleByCSS(parent);
    }

    #isElementInViewPort(element) {
        const rect = element.getBoundingClientRect();
        const viewport = {
            w: window.innerWidth,
            h: window.innerHeight
        };

        return rect.top < viewport.h && rect.bottom > 0 && rect.left < viewport.w && rect.right > 0;
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

        const w = this.#element.width;
        const h = this.#element.height;

        const iw = this.#image.width;
        const ih = this.#image.height;

        ctx.drawImage(this.#image, 0, 0, iw, ih, 0, 0, w, h);

        requestAnimationFrame(() => this.renderFrame());
    }

    #quote(name) {
        return '"' + name.replace(/"/, '\"') + '"';
    }
}