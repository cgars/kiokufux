const config = window.KIOKUFUX_GALLERY_CONFIG || { minTagCount: 2, maxCloudTags: 40 };
let data;
let all = [];
let shown = [];
let selectedTag = "";
let selectedIdentityId = "";
let currentIndex = 0;
let lastTrigger = null;
let touchStartX = 0;
let zoom = 1;
let fitWidth = 1;
let fitHeight = 1;
let faceBoxesVisible = false;
let panStart = null;

const queryInput = document.querySelector("#q");
const clearButton = document.querySelector("#clear");
const grid = document.querySelector("#grid");
const state = document.querySelector("#state");
const cloud = document.querySelector("#cloud");
const peopleFilter = document.querySelector("#people-filter");
const peopleCloud = document.querySelector("#people-cloud");
const empty = document.querySelector("#empty");
const box = document.querySelector("#box");
const fullImage = document.querySelector("#full");
const mediaViewport = document.querySelector("#media-viewport");
const mediaCanvas = document.querySelector("#media-canvas");
const faceOverlay = document.querySelector("#face-overlay");
const faceToggle = document.querySelector("#face-toggle");
const zoomLevel = document.querySelector("#zoom-level");
const caption = document.querySelector("#cap");
const description = document.querySelector("#description");
const descriptionToggle = document.querySelector("#description-toggle");
const position = document.querySelector("#position");
const detailTags = document.querySelector("#detail-tags");
const detailPeopleSection = document.querySelector("#detail-people-section");
const detailPeople = document.querySelector("#detail-people");
const details = document.querySelector("#details");
const lightboxInfo = document.querySelector(".lightbox-info");

const zoomSteps = [1, 1.25, 1.5, 2, 3, 4, 6];

function searchableText(item) {
  const people = (item.people || []).flatMap((person) => [person.label, person.display_name, person.friendly_name]);
  return [item.filename, item.relative_path, item.caption, item.description, ...(item.tags || []), ...people]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase()
    .replace(/\s+/g, " ");
}

function altText(item) {
  return item.caption || item.description || item.filename || "Archive photograph";
}

function displayDate(value) {
  if (!value) return "";
  const exif = value.match(/^(\d{4}):(\d{2}):(\d{2})(?:\s+(\d{2}):(\d{2}):(\d{2}))?/);
  const parsed = exif
    ? new Date(Number(exif[1]), Number(exif[2]) - 1, Number(exif[3]), Number(exif[4] || 0), Number(exif[5] || 0), Number(exif[6] || 0))
    : new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" }).format(parsed);
}

function setActiveTag(tag) {
  selectedTag = selectedTag === tag ? "" : tag;
  [...cloud.children].forEach((child) => {
    const active = child.dataset.tag === selectedTag;
    child.classList.toggle("active", active);
    child.setAttribute("aria-pressed", String(active));
  });
  applyFilters();
}

function identityKey(person) {
  return person.identity_id || (person.person_id ? `person:${person.person_id}` : "");
}

function identityStatusLabel(person) {
  if (person.status === "provisional") return "unconfirmed";
  if (person.status === "ungrouped") return "detected";
  return "";
}

function identityAvatar(person) {
  if (person.status === "ungrouped") return "?";
  return person.label.trim().charAt(0).toLocaleUpperCase() || "•";
}

function setActivePerson(identityId) {
  selectedIdentityId = selectedIdentityId === identityId ? "" : identityId;
  [...peopleCloud.children].forEach((child) => {
    const active = child.dataset.identityId === selectedIdentityId;
    child.classList.toggle("active", active);
    child.setAttribute("aria-pressed", String(active));
  });
  applyFilters();
}

function selectedPersonLabel() {
  return (data.people_frequencies || []).find((person) => identityKey(person) === selectedIdentityId)?.label || selectedIdentityId;
}

function makeCard(item, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "card";
  button.setAttribute("aria-label", `Open ${altText(item)}`);

  const image = document.createElement("img");
  image.loading = "lazy";
  image.decoding = "async";
  image.src = item.thumbnail_path;
  image.alt = "";
  button.appendChild(image);

  const copy = document.createElement("span");
  copy.className = "card-copy";

  const title = document.createElement("span");
  title.className = "card-title";
  title.textContent = item.caption || item.filename;
  copy.appendChild(title);

  const metadata = document.createElement("span");
  metadata.className = "card-meta";
  const previewPeople = (item.people || []).slice(0, 2).map((person) => person.label).join(" · ");
  const previewTags = (item.tags || []).slice(0, 2).join(" · ");
  metadata.textContent = [displayDate(item.datetime_original), previewPeople || previewTags].filter(Boolean).join("  —  ") || "View photograph";
  copy.appendChild(metadata);
  button.appendChild(copy);

  button.addEventListener("click", () => {
    lastTrigger = button;
    openLightbox(index);
  });
  return button;
}

function applyFilters() {
  const query = queryInput.value.trim().toLocaleLowerCase().replace(/\s+/g, " ");
  shown = all.filter((item) => {
    const tagMatches = !selectedTag || (item.tags || []).includes(selectedTag);
    const personMatches = !selectedIdentityId || (item.people || []).some((person) => identityKey(person) === selectedIdentityId);
    const queryMatches = !query || searchableText(item).includes(query);
    return tagMatches && personMatches && queryMatches;
  });

  const filters = [];
  if (query) filters.push(`matching “${query}”`);
  if (selectedTag) filters.push(`tagged “${selectedTag}”`);
  if (selectedIdentityId) filters.push(`with “${selectedPersonLabel()}”`);
  state.textContent = `${shown.length} ${shown.length === 1 ? "photograph" : "photographs"}${filters.length ? ` ${filters.join(" and ")}` : ` in this collection`}`;
  clearButton.hidden = !query && !selectedTag && !selectedIdentityId;
  empty.hidden = shown.length !== 0;
  grid.hidden = shown.length === 0;
  grid.replaceChildren(...shown.map(makeCard));
}

function buildTagCloud() {
  const entries = Object.entries(data.tag_frequencies || {})
    .filter((entry) => entry[1] >= config.minTagCount)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, config.maxCloudTags);

  entries.forEach(([tag, count]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.tag = tag;
    button.setAttribute("aria-pressed", "false");

    const label = document.createElement("span");
    label.textContent = tag;
    const amount = document.createElement("span");
    amount.className = "tag-count";
    amount.textContent = count;
    button.append(label, amount);
    button.addEventListener("click", () => setActiveTag(tag));
    cloud.appendChild(button);
  });
}

function buildPeopleCloud() {
  const people = data.people_frequencies || [];
  peopleFilter.hidden = people.length === 0;
  people.forEach((person) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `person-chip person-${person.status || "confirmed"}`;
    button.dataset.identityId = identityKey(person);
    button.setAttribute("aria-pressed", "false");
    const statusLabel = identityStatusLabel(person);
    button.title = statusLabel ? `${person.label} · ${statusLabel}` : person.label;

    const avatar = document.createElement("span");
    avatar.className = "person-avatar";
    avatar.textContent = identityAvatar(person);
    avatar.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.textContent = person.label;
    if (statusLabel) {
      const status = document.createElement("span");
      status.className = "person-status";
      status.textContent = statusLabel;
      label.appendChild(status);
    }
    const amount = document.createElement("span");
    amount.className = "tag-count";
    amount.textContent = person.count;
    button.append(avatar, label, amount);
    button.addEventListener("click", () => setActivePerson(identityKey(person)));
    peopleCloud.appendChild(button);
  });
}

function applyZoom(nextZoom, preserveCenter = true) {
  const previousWidth = mediaViewport.scrollWidth || mediaViewport.clientWidth;
  const previousHeight = mediaViewport.scrollHeight || mediaViewport.clientHeight;
  const centerX = (mediaViewport.scrollLeft + mediaViewport.clientWidth / 2) / previousWidth;
  const centerY = (mediaViewport.scrollTop + mediaViewport.clientHeight / 2) / previousHeight;
  zoom = Math.min(zoomSteps.at(-1), Math.max(zoomSteps[0], nextZoom));
  mediaCanvas.style.width = `${Math.max(1, Math.round(fitWidth * zoom))}px`;
  mediaCanvas.style.height = `${Math.max(1, Math.round(fitHeight * zoom))}px`;
  zoomLevel.value = `${Math.round(zoom * 100)}%`;
  zoomLevel.textContent = zoomLevel.value;

  requestAnimationFrame(() => {
    if (!preserveCenter || zoom === 1) {
      mediaViewport.scrollLeft = 0;
      mediaViewport.scrollTop = 0;
      return;
    }
    mediaViewport.scrollLeft = centerX * mediaViewport.scrollWidth - mediaViewport.clientWidth / 2;
    mediaViewport.scrollTop = centerY * mediaViewport.scrollHeight - mediaViewport.clientHeight / 2;
  });
}

function changeZoom(direction) {
  const currentStep = zoomSteps.findIndex((step) => step >= zoom - 0.001);
  const nextIndex = Math.min(zoomSteps.length - 1, Math.max(0, currentStep + direction));
  applyZoom(zoomSteps[nextIndex]);
}

function fitMedia() {
  if (!fullImage.naturalWidth || !fullImage.naturalHeight) return;
  const availableWidth = Math.max(1, mediaViewport.clientWidth);
  const availableHeight = Math.max(1, mediaViewport.clientHeight);
  const fitScale = Math.min(availableWidth / fullImage.naturalWidth, availableHeight / fullImage.naturalHeight);
  fitWidth = Math.max(1, fullImage.naturalWidth * fitScale);
  fitHeight = Math.max(1, fullImage.naturalHeight * fitScale);
  applyZoom(1, false);
}

function renderFaceBoxes(item) {
  const boxes = item.face_boxes || [];
  faceOverlay.replaceChildren(...boxes.map((face) => {
    const marker = document.createElement("span");
    marker.className = `face-box face-${face.status || "confirmed"}`;
    marker.style.left = `${face.box.x1 * 100}%`;
    marker.style.top = `${face.box.y1 * 100}%`;
    marker.style.width = `${(face.box.x2 - face.box.x1) * 100}%`;
    marker.style.height = `${(face.box.y2 - face.box.y1) * 100}%`;
    const label = document.createElement("span");
    label.className = "face-box-label";
    label.textContent = face.label;
    marker.appendChild(label);
    return marker;
  }));
  faceToggle.hidden = boxes.length === 0;
  faceOverlay.hidden = !faceBoxesVisible || boxes.length === 0;
  faceToggle.setAttribute("aria-pressed", String(faceBoxesVisible && boxes.length > 0));
  faceToggle.textContent = faceBoxesVisible ? "Hide faces" : "Show faces";
}

function setFaceBoxesVisible(visible) {
  faceBoxesVisible = visible;
  const item = shown[currentIndex];
  faceOverlay.hidden = !visible || !(item?.face_boxes || []).length;
  faceToggle.setAttribute("aria-pressed", String(visible));
  faceToggle.textContent = visible ? "Hide faces" : "Show faces";
}

function setDescription(value) {
  const text = value || "";
  description.textContent = text;
  const expandable = text.length > 320;
  description.classList.toggle("collapsed", expandable);
  descriptionToggle.hidden = !expandable;
  descriptionToggle.setAttribute("aria-expanded", "false");
  descriptionToggle.textContent = "Read full description";
}

function addDetail(label, value) {
  if (!value) return;
  const term = document.createElement("dt");
  term.textContent = label;
  const definition = document.createElement("dd");
  definition.textContent = value;
  details.append(term, definition);
}

function showItem(index) {
  currentIndex = index;
  const item = shown[currentIndex];
  if (!item) return;

  fullImage.src = item.image_path;
  fullImage.alt = altText(item);
  caption.textContent = item.caption || item.filename;
  setDescription(item.description);
  position.textContent = `Photograph ${currentIndex + 1} of ${shown.length}`;
  lightboxInfo.scrollTop = 0;
  renderFaceBoxes(item);
  if (fullImage.complete) requestAnimationFrame(fitMedia);

  detailPeople.replaceChildren();
  detailPeopleSection.hidden = !(item.people || []).length;
  (item.people || []).forEach((person) => {
    const button = document.createElement("button");
    button.type = "button";
    button.title = `Show photographs with ${person.label}`;
    button.className = `person-${person.status || "confirmed"}`;
    const avatar = document.createElement("span");
    avatar.className = "person-avatar";
    avatar.textContent = identityAvatar(person);
    avatar.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    const count = person.status === "ungrouped" && person.count_in_photo > 1 ? ` · ${person.count_in_photo}` : "";
    label.textContent = `${person.label}${count}`;
    const statusLabel = identityStatusLabel(person);
    if (statusLabel) {
      const status = document.createElement("span");
      status.className = "person-status";
      status.textContent = statusLabel;
      label.appendChild(status);
    }
    button.append(avatar, label);
    button.addEventListener("click", () => {
      box.close();
      selectedIdentityId = "";
      setActivePerson(identityKey(person));
      document.querySelector(".controls").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    detailPeople.appendChild(button);
  });

  detailTags.replaceChildren();
  (item.tags || []).forEach((tag) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = tag;
    button.title = `Filter gallery by ${tag}`;
    button.addEventListener("click", () => {
      box.close();
      selectedTag = "";
      setActiveTag(tag);
      document.querySelector(".controls").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    detailTags.appendChild(button);
  });

  details.replaceChildren();
  addDetail("Date", displayDate(item.datetime_original));
  addDetail("File", item.filename);
  if (item.dimensions?.width && item.dimensions?.height) {
    addDetail("Size", `${item.dimensions.width} × ${item.dimensions.height}`);
  }
  if (item.metadata?.gps?.lat != null && item.metadata?.gps?.lon != null) {
    addDetail("Location", `${Number(item.metadata.gps.lat).toFixed(5)}, ${Number(item.metadata.gps.lon).toFixed(5)}`);
  }
}

function openLightbox(index) {
  showItem(index);
  if (!box.open) {
    box.showModal();
    document.documentElement.classList.add("dialog-open");
    requestAnimationFrame(fitMedia);
  }
}

function moveLightbox(delta) {
  if (!shown.length) return;
  showItem((currentIndex + delta + shown.length) % shown.length);
}

function clearFilters() {
  queryInput.value = "";
  selectedTag = "";
  selectedIdentityId = "";
  [...cloud.children].forEach((child) => {
    child.classList.remove("active");
    child.setAttribute("aria-pressed", "false");
  });
  [...peopleCloud.children].forEach((child) => {
    child.classList.remove("active");
    child.setAttribute("aria-pressed", "false");
  });
  applyFilters();
  queryInput.focus();
}

clearButton.addEventListener("click", clearFilters);
document.querySelector("#empty-clear").addEventListener("click", clearFilters);
document.querySelector("#close").addEventListener("click", () => box.close());
document.querySelector("#prev").addEventListener("click", () => moveLightbox(-1));
document.querySelector("#next").addEventListener("click", () => moveLightbox(1));
document.querySelector("#zoom-out").addEventListener("click", () => changeZoom(-1));
document.querySelector("#zoom-in").addEventListener("click", () => changeZoom(1));
document.querySelector("#zoom-reset").addEventListener("click", fitMedia);
faceToggle.addEventListener("click", () => setFaceBoxesVisible(!faceBoxesVisible));
descriptionToggle.addEventListener("click", () => {
  const expanded = descriptionToggle.getAttribute("aria-expanded") === "true";
  description.classList.toggle("collapsed", expanded);
  descriptionToggle.setAttribute("aria-expanded", String(!expanded));
  descriptionToggle.textContent = expanded ? "Read full description" : "Show less";
});
fullImage.addEventListener("load", fitMedia);
document.querySelector("#density").addEventListener("click", (event) => {
  const compact = grid.classList.toggle("compact");
  event.currentTarget.setAttribute("aria-pressed", String(compact));
  event.currentTarget.title = compact ? "Show larger photos" : "Show smaller photos";
});

box.addEventListener("click", (event) => {
  if (event.target === box) box.close();
});
box.addEventListener("close", () => {
  document.documentElement.classList.remove("dialog-open");
  lastTrigger?.focus();
});
mediaViewport.addEventListener("touchstart", (event) => {
  touchStartX = event.changedTouches[0].clientX;
}, { passive: true });
mediaViewport.addEventListener("touchend", (event) => {
  const distance = event.changedTouches[0].clientX - touchStartX;
  if (zoom === 1 && Math.abs(distance) > 60) moveLightbox(distance > 0 ? -1 : 1);
}, { passive: true });

mediaViewport.addEventListener("dblclick", () => applyZoom(zoom === 1 ? 2 : 1));
mediaViewport.addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  changeZoom(event.deltaY < 0 ? 1 : -1);
}, { passive: false });
mediaViewport.addEventListener("pointerdown", (event) => {
  if (zoom === 1 || event.button !== 0) return;
  panStart = { x: event.clientX, y: event.clientY, left: mediaViewport.scrollLeft, top: mediaViewport.scrollTop };
  mediaViewport.classList.add("dragging");
  mediaViewport.setPointerCapture(event.pointerId);
});
mediaViewport.addEventListener("pointermove", (event) => {
  if (!panStart) return;
  mediaViewport.scrollLeft = panStart.left - (event.clientX - panStart.x);
  mediaViewport.scrollTop = panStart.top - (event.clientY - panStart.y);
});
function stopPanning(event) {
  if (!panStart) return;
  panStart = null;
  mediaViewport.classList.remove("dragging");
  if (mediaViewport.hasPointerCapture(event.pointerId)) mediaViewport.releasePointerCapture(event.pointerId);
}
mediaViewport.addEventListener("pointerup", stopPanning);
mediaViewport.addEventListener("pointercancel", stopPanning);

document.addEventListener("keydown", (event) => {
  if (box.open && zoom > 1 && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
    event.preventDefault();
    const directions = { ArrowLeft: [-80, 0], ArrowRight: [80, 0], ArrowUp: [0, -80], ArrowDown: [0, 80] };
    mediaViewport.scrollBy(...directions[event.key]);
  } else if (box.open && event.key === "ArrowLeft") moveLightbox(-1);
  else if (box.open && event.key === "ArrowRight") moveLightbox(1);
  if (box.open && ["+", "="].includes(event.key)) changeZoom(1);
  if (box.open && event.key === "-") changeZoom(-1);
  if (box.open && event.key === "0") fitMedia();
  if (!box.open && event.key === "/" && document.activeElement !== queryInput) {
    event.preventDefault();
    queryInput.focus();
  }
});
queryInput.addEventListener("input", applyFilters);
window.addEventListener("resize", () => {
  if (box.open) fitMedia();
});

data = JSON.parse(document.querySelector("#gallery-data").textContent);
all = Array.isArray(data.items) ? data.items : [];
buildTagCloud();
buildPeopleCloud();
applyFilters();
