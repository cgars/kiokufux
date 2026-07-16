const config = window.KIOKUFUX_GALLERY_CONFIG || { minTagCount: 2, maxCloudTags: 40 };
let data;
let all = [];
let shown = [];
let selectedTag = "";
let currentIndex = 0;

const queryInput = document.querySelector("#q");
const grid = document.querySelector("#grid");
const state = document.querySelector("#state");
const cloud = document.querySelector("#cloud");
const empty = document.querySelector("#empty");
const box = document.querySelector("#box");

function searchableText(item) {
  return [item.filename, item.relative_path, item.caption, item.description, ...(item.tags || [])]
    .join(" ")
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function altText(item) {
  return item.caption || item.description || item.filename;
}

function applyFilters() {
  const query = queryInput.value.trim().toLowerCase().replace(/\s+/g, " ");
  shown = all.filter((item) => {
    const tagMatches = !selectedTag || item.tags.includes(selectedTag);
    const queryMatches = !query || searchableText(item).includes(query);
    return tagMatches && queryMatches;
  });

  state.textContent = `${shown.length} of ${all.length} photos${query ? ` · search: "${query}"` : ""}${selectedTag ? ` · tag: ${selectedTag}` : ""}`;
  empty.hidden = shown.length !== 0;
  grid.innerHTML = "";
  shown.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "card";

    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = item.thumbnail_path;
    image.alt = altText(item);
    button.appendChild(image);

    const label = document.createElement("span");
    label.textContent = item.caption || item.filename;
    button.appendChild(label);

    button.addEventListener("click", () => openLightbox(index));
    grid.appendChild(button);
  });
}

function buildTagCloud() {
  const entries = Object.entries(data.tag_frequencies)
    .filter((entry) => entry[1] >= config.minTagCount)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, config.maxCloudTags);
  const maxCount = Math.max(1, ...entries.map((entry) => entry[1]));

  entries.forEach(([tag, count]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${tag} · ${count}`;
    button.style.fontSize = `${0.9 + (Math.log(count + 1) / Math.log(maxCount + 1)) * 0.9}rem`;
    button.addEventListener("click", () => {
      selectedTag = selectedTag === tag ? "" : tag;
      [...cloud.children].forEach((child) => child.classList.toggle("active", child === button && Boolean(selectedTag)));
      applyFilters();
    });
    cloud.appendChild(button);
  });
}

function openLightbox(index) {
  currentIndex = index;
  const item = shown[currentIndex];
  document.querySelector("#full").src = item.image_path;
  document.querySelector("#full").alt = altText(item);
  document.querySelector("#cap").textContent = item.caption || item.filename;
  document.querySelector("#detail").textContent = [item.description, (item.tags || []).join(", "), item.datetime_original]
    .filter(Boolean)
    .join(" · ");
  box.showModal();
}

function moveLightbox(delta) {
  if (!shown.length) return;
  openLightbox((currentIndex + delta + shown.length) % shown.length);
}

document.querySelector("#clear").addEventListener("click", () => {
  queryInput.value = "";
  selectedTag = "";
  [...cloud.children].forEach((child) => child.classList.remove("active"));
  applyFilters();
});
document.querySelector("#close").addEventListener("click", () => box.close());
document.querySelector("#prev").addEventListener("click", () => moveLightbox(-1));
document.querySelector("#next").addEventListener("click", () => moveLightbox(1));
document.addEventListener("keydown", (event) => {
  if (box.open && event.key === "ArrowLeft") moveLightbox(-1);
  if (box.open && event.key === "ArrowRight") moveLightbox(1);
});
queryInput.addEventListener("input", applyFilters);

data = JSON.parse(document.querySelector("#gallery-data").textContent);
all = data.items;
buildTagCloud();
applyFilters();
