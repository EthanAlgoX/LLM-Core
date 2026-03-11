import DefaultTheme from "vitepress/theme";
import { defineComponent, h, nextTick, onMounted, watch } from "vue";
import { useRoute } from "vitepress";

import "./custom.css";

const Layout = defineComponent({
  name: "LLMCoreThemeLayout",
  setup() {
    const route = useRoute();

    const renderMermaid = async () => {
      if (typeof window === "undefined") {
        return;
      }

      await nextTick();

      const mermaid = (await import("mermaid")).default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: document.documentElement.classList.contains("dark") ? "dark" : "neutral",
      });
      await mermaid.run({
        querySelector: ".mermaid",
      });
    };

    onMounted(() => {
      void renderMermaid();
    });

    watch(
      () => route.path,
      () => {
        void renderMermaid();
      },
    );

    return () => h(DefaultTheme.Layout);
  },
});

export default {
  extends: DefaultTheme,
  Layout,
};
