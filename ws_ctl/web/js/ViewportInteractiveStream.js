import ViewportStream from "./ViewportStream.js";

export default class ViewportInteractiveStream extends ViewportStream {
    #ctx;
    #infoImageName;

    constructor(canvas, socket) {
        super(canvas, socket);
        this.#infoImageName = canvas.dataset.infoImage;
    }

    onImage(event) {
        if (event.detail.header.name !== this.#infoImageName) {
            return;
        }
        super.onImage(event);
    }

    renderFrame() {
        super.renderFrame();
    }
}