import Telemetry from "./Telemetry.js";

export default class Logger {
    #dbName;
    #storeName;
    #db;
    #container;
    constructor(element) {
        this.#dbName = 'ecv.Logger'
        this.#storeName = 'ecv.Logger.Log'

        this.#initDb().finally(async () => {
            this.#container = element.getElementsByClassName("log-box").item(0);
            for (const row of await this.#getLogs()) {
                this.#addUiItem(row.timestamp, row.level, row.message, 1000);
            }
            document.addEventListener(Telemetry.EVENT_NAME_TELEMETRY, (event) => this.onTelemetry(event.detail));

            this.#cleanupOldRecords(1000);
        });
    }

    #addUiItem(ts, level, message, limit) {
        const container = this.#container;
        if (!container) return;

        // Проверяем, находится ли скролл внизу (с допуском в 5 пикселей)
        const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 5;

        const date = ts ? new Date(ts) : new Date();
        const timeStr = date.toLocaleTimeString('ru-RU', { hour12: false });

        const entry = document.createElement('div');
        entry.className = `log-entry log-entry-${level.toLowerCase()}`;
        entry.innerHTML = `<span class="log-row log-level-${level.toLowerCase()}"><span class="log-time">[${timeStr}]</span>: ${message}</span>`;

        container.append(entry);

        // Ограничиваем количество записей (удаляем самые старые – первые)
        while (container.children.length > limit) {
            container.removeChild(container.firstChild);
        }

        // Если до добавления были внизу – прокручиваем в самый низ после всех изменений
        if (isAtBottom) {
            container.scrollTop = container.scrollHeight;
        }
    }

    async #initDb() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.#dbName, 1);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                // Создаём хранилище с auto-increment ключом
                const store = db.createObjectStore(this.#storeName, {
                    keyPath: 'id',
                    autoIncrement: true
                });
                // Создаём индексы для удобного поиска (опционально)
                store.createIndex('timestamp', 'timestamp');
            };

            request.onsuccess = (event) => {
                this.#db = event.target.result;
                resolve();
            };

            request.onerror = (event) => {
                reject(event.target.error);
            };
        });
    }

    onTelemetry(event) {
        if (event.command !== 'LOG') {
            return;
        }

        const level = event.args.shift();
        const ts = parseFloat(event.args.shift());
        const message = event.args.shift();

        this.#addUiItem(ts, level, message, 1000);
        this.#addDbItem(ts, level, message, 1000).then(() => {});
    }

    log(level, message) {
        const ts = Date.now();
        this.#addUiItem(ts, level, message, 1000);
        this.#addDbItem(ts, level, message, 1000).then(() => {});
    }

    debug(message) {
        this.log('debug', message);
    }

    info(message) {
        this.log('info', message);
    }

    warn(message) {
        this.log('warn', message);
    }

    error(message) {
        this.log('error', message);
    }

    async #addDbItem(ts, level, message, limit) {
        if (!this.#db) {
            console.warn('База данных не инициализирована');
            return;
        }

        const record = {
            timestamp: ts,
            level: level,
            message: message
        };

        const id = await this.#addRecord(record);

        this.#cleanupOldRecords(limit).catch(err =>
            console.warn('Ошибка при фоновой чистке логов:', err)
        );

        return id;
    }

    #addRecord(record) {
        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction([this.#storeName], 'readwrite');
            const store = transaction.objectStore(this.#storeName);
            const request = store.add(record);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async #getLogs() {
        if (!this.#db) {
            console.warn('База данных не инициализирована');
            return;
        }

        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction([this.#storeName], 'readonly');
            const store = transaction.objectStore(this.#storeName);
            const index = store.index('timestamp');
            const request = index.openCursor(); // по умолчанию ASC (от старых к новым)

            const results = [];
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    results.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };
            request.onerror = (event) => reject(event.target.error);
        });
    }

    async #cleanupOldRecords(limit) {
        // Получаем общее количество записей
        const total = await this.#countRecords();
        if (total <= limit) return;

        const toDelete = total - limit;
        const transaction = this.#db.transaction([this.#storeName], 'readwrite');
        const store = transaction.objectStore(this.#storeName);
        const index = store.index('timestamp');

        await new Promise((resolve, reject) => {
            const cursorRequest = index.openCursor(); // от старых к новым
            let deleted = 0;

            cursorRequest.onsuccess = (event) => {
                const cursor = event.target.result;
                if (!cursor || deleted >= toDelete) {
                    resolve();
                    return;
                }

                const deleteRequest = cursor.delete();
                deleteRequest.onsuccess = () => {
                    deleted++;
                    cursor.continue();
                };
                deleteRequest.onerror = (err) => reject(err);
            };
            cursorRequest.onerror = (err) => reject(err);
        });
    }

    #countRecords() {
        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction([this.#storeName], 'readonly');
            const store = transaction.objectStore(this.#storeName);
            const request = store.count();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
}