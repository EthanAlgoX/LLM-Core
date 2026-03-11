import fs from "node:fs/promises";
import path from "node:path";

import * as cheerio from "cheerio";

import { siteConfig } from "@/lib/site-config";

export type DocRecord = {
  slug: string[];
  slugPath: string;
  sourceFile: string;
  title: string;
  summary: string;
  groupKey: string;
  groupLabel: string;
  href: string;
  publicAccess: boolean;
  isOverview: boolean;
  bodyHtml: string;
  previewHtml: string;
  outline: Array<{ id: string; text: string; level: 2 | 3 }>;
};

let docManifestCache: DocRecord[] | null = null;

function resolveContentRoot() {
  const candidates = [
    process.env.CONTENT_HTML_ROOT,
    path.resolve(process.cwd(), "../../html"),
    path.resolve(process.cwd(), "../html"),
  ].filter(Boolean) as string[];

  return candidates[0];
}

async function collectHtmlFiles(rootDir: string, currentDir = rootDir): Promise<string[]> {
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    if (entry.name === "_site" || entry.name.startsWith(".")) {
      continue;
    }

    const absolutePath = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectHtmlFiles(rootDir, absolutePath)));
      continue;
    }

    if (!entry.isFile() || !entry.name.endsWith(".html")) {
      continue;
    }

    files.push(path.relative(rootDir, absolutePath));
  }

  return files;
}

function deriveGroupKey(relativePath: string) {
  if (relativePath === "README.html") {
    return "home";
  }

  const [topLevel, second] = relativePath.split(path.sep);
  if (topLevel === "docs") return "guides";
  if (topLevel === "modules" && second) return second;
  if (topLevel === "output") return "reports";
  return topLevel;
}

function deriveGroupLabel(groupKey: string) {
  const labels: Record<string, string> = {
    home: "Overview",
    guides: "Guides",
    "01_foundation_rl": "RL Foundation",
    "02_architecture": "Architecture",
    "03_alignment": "Alignment",
    "04_advanced_topics": "Advanced Topics",
    "05_engineering": "Engineering",
    "06_agent": "Agent",
    "07_classic_models": "Classic Models",
    reports: "Reports",
  };
  return labels[groupKey] ?? groupKey;
}

function isOverviewDoc(relativePath: string) {
  if (relativePath === "README.html") {
    return true;
  }

  const parsed = path.parse(relativePath);
  return parsed.name === path.basename(parsed.dir) || parsed.base === "README.html";
}

function makePreviewHtml(bodyHtml: string) {
  const $ = cheerio.load(`<article>${bodyHtml}</article>`);
  const article = $("article");
  const preview = article.children().slice(0, 8).toArray().map((node) => $.html(node)).join("");
  return preview || bodyHtml;
}

function buildOutline(bodyHtml: string) {
  const $ = cheerio.load(`<article>${bodyHtml}</article>`);
  return $("h2[id], h3[id]")
    .toArray()
    .map((node) => {
      const el = $(node);
      el.find(".heading-anchor").remove();
      return {
        id: el.attr("id") ?? "",
        text: el.text().trim(),
        level: node.tagName === "h2" ? (2 as const) : (3 as const),
      };
    })
    .filter((item) => item.id && item.text);
}

async function parseDoc(relativePath: string, contentRoot: string) {
  const absolutePath = path.join(contentRoot, relativePath);
  const html = await fs.readFile(absolutePath, "utf-8");
  const $ = cheerio.load(html);
  const title = $("title").text().trim() || path.parse(relativePath).name;
  const summary = $('meta[name="description"]').attr("content")?.trim() || "";
  const bodyHtml = $(".article-content").html()?.trim() || "";
  const slugPath = relativePath === "README.html" ? "" : relativePath.replace(/\.html$/, "").split(path.sep).join("/");

  return {
    slug: slugPath ? slugPath.split("/") : [],
    slugPath,
    sourceFile: relativePath,
    title,
    summary,
    bodyHtml,
    previewHtml: makePreviewHtml(bodyHtml),
    outline: buildOutline(bodyHtml),
    groupKey: deriveGroupKey(relativePath),
    groupLabel: deriveGroupLabel(deriveGroupKey(relativePath)),
    href: slugPath ? `/docs/${slugPath}` : "/",
    isOverview: isOverviewDoc(relativePath),
  };
}

function applyPublicAccessRules(docs: DocRecord[]) {
  const previewCounters = new Map<string, number>();

  for (const doc of docs) {
    if (doc.slugPath === "") {
      doc.publicAccess = true;
      continue;
    }

    if (doc.groupKey === "guides" || doc.groupKey === "reports") {
      doc.publicAccess = true;
      continue;
    }

    if (doc.isOverview) {
      doc.publicAccess = true;
      continue;
    }

    const seen = previewCounters.get(doc.groupKey) ?? 0;
    if (seen < siteConfig.previewDocsPerModule) {
      doc.publicAccess = true;
      previewCounters.set(doc.groupKey, seen + 1);
      continue;
    }

    doc.publicAccess = false;
  }
}

export async function getDocManifest() {
  if (docManifestCache) {
    return docManifestCache;
  }

  const contentRoot = resolveContentRoot();
  const files = await collectHtmlFiles(contentRoot);
  const docs = await Promise.all(files.sort().map((relativePath) => parseDoc(relativePath, contentRoot)));
  applyPublicAccessRules(docs as DocRecord[]);
  docManifestCache = docs as DocRecord[];
  return docManifestCache;
}

export async function getDocBySlug(slug: string[]) {
  const docs = await getDocManifest();
  return docs.find((doc) => doc.slugPath === slug.join("/")) ?? null;
}

export async function getGroupedDocs() {
  const docs = await getDocManifest();
  const groups = new Map<string, DocRecord[]>();
  for (const doc of docs) {
    const key = doc.groupKey;
    const list = groups.get(key) ?? [];
    list.push(doc);
    groups.set(key, list);
  }
  return groups;
}

export function clearDocManifestCache() {
  docManifestCache = null;
}
