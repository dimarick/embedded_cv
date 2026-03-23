import ViewportStream from "./ViewportStream.js";
import ViewportInteractiveStream from "./ViewportInteractiveStream.js";

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
            if (element.classList.contains('interactive-viewport')) {
                const valueElement = element.getElementsByClassName('interactive-viewport-value').item(0)
                this.#steam = new ViewportInteractiveStream(canvas, sockets, valueElement);
                continue;
            }
            this.#steam = new ViewportStream(canvas, sockets);
        }
    }

    getTitle() {
        return this.#title.textContent;
    }
}