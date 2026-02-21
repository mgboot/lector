const STORAGE_KEY = "lector-words";

function _getAll() {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw ? JSON.parse(raw) : {};
}

function _saveAll(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function hasWord(word) {
  return word in _getAll();
}

export function getWord(word) {
  return _getAll()[word] ?? null;
}

export function setWord(word, data) {
  const all = _getAll();
  all[word] = data;
  _saveAll(all);
}

export function getAllWords() {
  return _getAll();
}
