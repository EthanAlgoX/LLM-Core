import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const siteData = window.__LLM_CORE_SITE__ || { pages: [] };
const currentPath = window.__LLM_CORE_PAGE__?.path || "README.html";

const article = document.querySelector(".article-content");
const sidebar = document.getElementById("site-sidebar");
const navToggle = document.getElementById("nav-toggle");
const searchButton = document.getElementById("search-button");
const themeToggle = document.getElementById("theme-toggle");
const searchInput = document.getElementById("site-search-input");
const searchResults = document.getElementById("search-results");
const tocNav = document.getElementById("toc-nav");
const progressBar = document.getElementById("scroll-progress-bar");

mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });

function normalizeParts(path) {
  return path.split("/").filter(Boolean);
}

function toRelativeHref(targetPath) {
  const fromParts = normalizeParts(currentPath).slice(0, -1);
  const toParts = normalizeParts(targetPath);
  let index = 0;
  while (index < fromParts.length && index < toParts.length && fromParts[index] === toParts[index]) {
    index += 1;
  }

  const up = new Array(fromParts.length - index).fill("..");
  const down = toParts.slice(index);
  const href = [...up, ...down].join("/");
  return href || "./";
}

function focusSearch() {
  if (window.innerWidth <= 980) {
    document.body.classList.add("sidebar-open");
    navToggle?.setAttribute("aria-expanded", "true");
  }
  searchInput?.focus();
  searchInput?.select();
}

function getTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("llm-core-theme", theme);
  } catch (_error) {}
  if (!themeToggle) {
    return;
  }
  const dark = theme === "dark";
  themeToggle.textContent = dark ? "Light" : "Dark";
  themeToggle.setAttribute("aria-pressed", String(dark));
}

function renderMath() {
  if (!window.renderMathInElement || !article) {
    return;
  }
  window.renderMathInElement(article, {
    throwOnError: false,
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
}

async function renderMermaid() {
  const blocks = [...document.querySelectorAll("pre code.language-mermaid")];
  for (const code of blocks) {
    const wrapper = document.createElement("div");
    wrapper.className = "mermaid";
    wrapper.textContent = code.textContent;
    code.parentElement.replaceWith(wrapper);
  }

  if (document.querySelector(".mermaid")) {
    await mermaid.run({ querySelector: ".mermaid" });
  }
}

function updateScrollProgress() {
  if (!progressBar) {
    return;
  }
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  const progress = maxScroll > 0 ? (window.scrollY / maxScroll) * 100 : 0;
  progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
}

function buildToc() {
  if (!tocNav || !article) {
    return;
  }

  const headings = [...article.querySelectorAll("h2[id], h3[id]")];
  if (!headings.length) {
    tocNav.innerHTML = '<p class="toc-empty">No headings on this page.</p>';
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const heading of headings) {
    const source = heading.cloneNode(true);
    source.querySelector(".heading-anchor")?.remove();

    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.className = `toc-link depth-${heading.tagName === "H2" ? "2" : "3"}`;
    link.textContent = source.textContent.trim();
    fragment.appendChild(link);
  }

  tocNav.innerHTML = "";
  tocNav.appendChild(fragment);

  const links = [...tocNav.querySelectorAll(".toc-link")];
  const byId = new Map(links.map((link) => [link.getAttribute("href")?.slice(1), link]));
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const link = byId.get(entry.target.id);
        if (!link || !entry.isIntersecting) {
          continue;
        }
        links.forEach((item) => item.classList.remove("active"));
        link.classList.add("active");
      }
    },
    {
      rootMargin: "-20% 0px -65% 0px",
      threshold: 0.08,
    },
  );

  headings.forEach((heading) => observer.observe(heading));
}

function searchMatches(query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return [];
  }
  const terms = normalized.split(/\s+/).filter(Boolean);

  return siteData.pages
    .map((page) => {
      const title = page.title.toLowerCase();
      const summary = page.summary.toLowerCase();
      const group = page.group.toLowerCase();
      const source = page.source.toLowerCase();
      const haystack = `${title} ${summary} ${group} ${source}`;
      let score = 0;

      if (title === normalized) score += 160;
      if (title.startsWith(normalized)) score += 110;
      if (title.includes(normalized)) score += 70;
      if (group.includes(normalized)) score += 28;
      if (source.includes(normalized)) score += 18;
      if (page.is_overview) score += 6;

      for (const term of terms) {
        if (title.startsWith(term)) score += 22;
        if (title.includes(term)) score += 14;
        if (summary.includes(term)) score += 6;
        if (group.includes(term)) score += 4;
        if (!haystack.includes(term)) score -= 24;
      }

      return { ...page, score };
    })
    .filter((page) => page.score > 0)
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      if (left.order && right.order) {
        for (let index = 0; index < Math.min(left.order.length, right.order.length); index += 1) {
          if (left.order[index] === right.order[index]) continue;
          if (typeof left.order[index] === "number" && typeof right.order[index] === "number") {
            return left.order[index] - right.order[index];
          }
          return String(left.order[index]).localeCompare(String(right.order[index]), "zh-CN");
        }
      }
      return left.title.localeCompare(right.title, "zh-CN");
    })
    .slice(0, 8);
}

function renderSearchResults(query) {
  if (!searchResults) {
    return;
  }

  const normalized = query.trim();
  if (!normalized) {
    searchResults.innerHTML =
      '<div class="search-empty"><span>Tips</span><p>Use "/" or the Search button to jump between modules, guides, and reports.</p></div>';
    return;
  }

  const matches = searchMatches(normalized);
  if (!matches.length) {
    searchResults.innerHTML =
      '<div class="search-empty"><span>No Match</span><p>Try a module name, method, or concept keyword.</p></div>';
    return;
  }

  searchResults.innerHTML = matches
    .map((page) => {
      const href = toRelativeHref(page.path);
      return `
        <a class="search-result" href="${href}">
          <span>${page.group}</span>
          <strong>${page.title}</strong>
          <p>${page.summary}</p>
        </a>
      `;
    })
    .join("");
}

function bindEvents() {
  navToggle?.addEventListener("click", () => {
    const isOpen = document.body.classList.toggle("sidebar-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });

  searchButton?.addEventListener("click", focusSearch);

  themeToggle?.addEventListener("click", () => {
    applyTheme(getTheme() === "dark" ? "light" : "dark");
  });

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof HTMLElement)) {
      return;
    }

    if (
      document.body.classList.contains("sidebar-open") &&
      !sidebar?.contains(event.target) &&
      !navToggle?.contains(event.target)
    ) {
      document.body.classList.remove("sidebar-open");
      navToggle?.setAttribute("aria-expanded", "false");
    }
  });

  document.addEventListener("keydown", (event) => {
    if ((event.key === "k" && (event.metaKey || event.ctrlKey)) || event.key === "/") {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      event.preventDefault();
      focusSearch();
    }

    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
      document.body.classList.remove("sidebar-open");
      navToggle?.setAttribute("aria-expanded", "false");
    }
  });

  searchInput?.addEventListener("input", (event) => {
    renderSearchResults(event.currentTarget.value);
  });

  window.addEventListener("scroll", updateScrollProgress, { passive: true });
}

async function init() {
  applyTheme(getTheme());
  renderMath();
  await renderMermaid();
  buildToc();
  renderSearchResults("");
  updateScrollProgress();
  bindEvents();
}

window.addEventListener("DOMContentLoaded", init);
