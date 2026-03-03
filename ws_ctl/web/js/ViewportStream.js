import Socket from "./Socket";

export default class ViewportStream {
    #element;
    #socket = null;
    #w = 0;
    #h = 0;

    constructor(element) {
        this.#element = element;
        this.#socket = new Socket('/' + this.#host + '?view=' + encodeURIComponent(this.#element.dataset.name), this.onMessage);
        this.resize();
        setInterval(() => {
            this.resize();
        }, 1000);
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

    onMessage(data) {
        const imageData = this.parseData(data);
    }

    parseData(data) {
        const buffer = data.arrayBuffer();
        const view = new DataView(buffer);

        let offset = 0;
        const type = view.getUint8(offset); offset += 1;        // CvMatTypeEnum
        const codec = view.getUint8(offset); offset += 1;       // CvMatCodecEnum
        const channels = view.getUint8(offset); offset += 1;    // число каналов
        const x = view.getUint16(offset, true); offset += 2;
        const y = view.getUint16(offset, true); offset += 2;
        const w = view.getUint16(offset, true); offset += 2;
        const h = view.getUint16(offset, true); offset += 2;
        const scale = view.getFloat32(offset, true); offset += 4;

        // Определяем количество байт на один канал по типу данных
        const typeToBytesPerChannel = {
            0x10: 1, // TYPE_8U
            0x11: 1, // TYPE_8S
            0x20: 2, // TYPE_16U
            0x21: 2, // TYPE_16S
            0x22: 2, // TYPE_16F
            0x32: 4, // TYPE_32F
            0x42: 8, // TYPE_64F
        };
        const bytesPerChannel = typeToBytesPerChannel[type];
        if (bytesPerChannel === undefined) {
            throw new Error(`Unsupported image type: 0x${type.toString(16)}`);
        }

        // Проверяем размер данных изображения
        const expectedDataSize = w * h * channels * bytesPerChannel;
        const availableSize = buffer.byteLength - offset;
        if (availableSize < expectedDataSize) {
            throw new Error(
                `Incomplete image data: expected ${expectedDataSize} bytes, ` +
                `got ${availableSize}`
            );
        }

        // Извлекаем сырые пиксели (ArrayBuffer)
        const pixelData = buffer.slice(offset, offset + expectedDataSize);

        // Возвращаем структурированный результат
        return {
            header: {
                type,
                codec,
                channels,
                x,
                y,
                w,
                h,
                scale,
            },
            pixelData,
        };
    }
}
function fourCC(str) {
    if (str.length !== 4) throw new Error('FourCC must be 4 chars');
    return (str.charCodeAt(0) << 24) |
        (str.charCodeAt(1) << 16) |
        (str.charCodeAt(2) << 8)  |
        str.charCodeAt(3);
}
/**
 * Преобразует объект с данными в бинарный ArrayBuffer в соответствии со схемой типов.
 * @param {Object} data - объект, где ключи соответствуют полям схемы, значения - числа
 * @param {Object} schema - объект, где ключи - имена полей, значения - строки типа ('int', 'uint', 'short', 'ushort', 'char', 'uchar', 'float', 'double')
 * @returns {ArrayBuffer} - бинарное представление данных (little-endian)
 */
function toBinaryStruct(data, schema) {
    // Размеры типов в байтах
    const typeSizes = {
        'int': 4,
        'uint': 4,
        'short': 2,
        'ushort': 2,
        'char': 1,
        'uchar': 1,
        'float': 4,
        'double': 8
    };

    // Вычисляем общий размер и собираем информацию о полях в порядке схемы
    let totalSize = 0;
    const fields = [];
    for (const key in schema) {
        if (schema.hasOwnProperty(key)) {
            const type = schema[key];
            const size = typeSizes[type];
            if (size === undefined) {
                throw new Error(`Неподдерживаемый тип: ${type}`);
            }
            fields.push({ key, type, offset: totalSize });
            totalSize += size;
        }
    }

    // Создаём буфер и DataView
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // Записываем каждое поле
    for (const field of fields) {
        const value = data[field.key];
        if (typeof value !== 'number') {
            throw new Error(`Поле ${field.key} должно быть числом`);
        }

        switch (field.type) {
            case 'int':
                view.setInt32(field.offset, value, true);
                break;
            case 'uint':
                view.setUint32(field.offset, value, true);
                break;
            case 'short':
                view.setInt16(field.offset, value, true);
                break;
            case 'ushort':
                view.setUint16(field.offset, value, true);
                break;
            case 'char':
                view.setInt8(field.offset, value);
                break;
            case 'uchar':
                view.setUint8(field.offset, value);
                break;
            case 'float':
                view.setFloat32(field.offset, value, true);
                break;
            case 'double':
                view.setFloat64(field.offset, value, true);
                break;
            default:
                throw new Error(`Неподдерживаемый тип: ${field.type}`);
        }
    }

    return buffer;
}