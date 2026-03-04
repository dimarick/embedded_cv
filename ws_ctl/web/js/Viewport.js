import ViewportStream from "./ViewportStream";

export default class Viewport {
    #element;
    #title;
    #steam;
    constructor(element, socket) {
        this.#element = element;
        this.#title = element.getElementsByClassName("viewport-title");
        this.#steam = new ViewportStream(element.getElementsByClassName("viewport-canvas-stream"), socket);
    }

    getTitle() {
        return this.#title.textContent;
    }
}