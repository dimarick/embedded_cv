import ViewportStream from "./ViewportStream.js";

export default class Viewport {
    #element;
    #title;
    #steam;
    constructor(element, sockets) {
        this.#element = element;
        this.#title = element.getElementsByClassName("viewport-title");
        for (const canvas of element.getElementsByClassName("viewport-canvas-stream")) {
            if (canvas.dataset === undefined || !canvas.dataset.name) {
                 continue;
            }
            this.#steam = new ViewportStream(canvas, sockets);
        }
    }

    getTitle() {
        return this.#title.textContent;
    }
}