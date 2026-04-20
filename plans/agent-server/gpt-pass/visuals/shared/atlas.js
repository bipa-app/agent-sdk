(function () {
  const PAGE_LIBRARY = [
    {
      title: "Sequence Views",
      items: [
        {
          id: "root-turn-flow",
          title: "Root Turn Flow",
          href: "sequence/01-root-turn-flow.html",
          copy: "One root turn from acquisition through commit, suspension, and terminal outcomes.",
        },
        {
          id: "tool-runtime-flow",
          title: "Tool Runtime Flow",
          href: "sequence/02-tool-runtime-flow.html",
          copy: "How root turns suspend, create child tasks, and resume from durable tool results.",
        },
        {
          id: "checkpoint-recovery-flow",
          title: "Checkpoint Recovery Flow",
          href: "sequence/03-checkpoint-recovery-flow.html",
          copy: "What recovery loads after clean commit, pre-commit crash, and post-commit crash.",
        },
        {
          id: "event-replay-flow",
          title: "Event Replay Flow",
          href: "sequence/04-event-replay-flow.html",
          copy: "Committed event ordering, replay handoff, and lagged subscriber recovery.",
        },
        {
          id: "subagent-thread-flow",
          title: "Subagent Thread Flow",
          href: "sequence/05-subagent-thread-flow.html",
          copy: "Durable child thread creation, summary progress, and terminal parent resolution.",
        },
      ],
    },
    {
      title: "State Views",
      items: [
        {
          id: "root-task-state-machine",
          title: "Root Task State Machine",
          href: "state/01-root-task-state-machine.html",
          copy: "The authoritative lifecycle for root work across turns, children, and confirmation.",
        },
        {
          id: "tool-task-state-machine",
          title: "Tool Task State Machine",
          href: "state/02-tool-task-state-machine.html",
          copy: "Tool child task lifecycle with confirmation, failure, and replay-safe restart rules.",
        },
        {
          id: "authority-commit-map",
          title: "Authority And Commit Map",
          href: "state/03-authority-and-commit-map.html",
          copy: "Which subsystem owns each decision and where durable commit boundaries sit.",
        },
        {
          id: "schema-relationship-map",
          title: "Schema Relationship Map",
          href: "state/04-schema-relationship-map.html",
          copy: "Core tables, constraints, and recovery-critical relationships in the final model.",
        },
      ],
    },
  ];

  const TONE_COLORS = {
    contracts: "#ca6f37",
    journal: "#9c7320",
    runtime: "#45637f",
    persistence: "#3a7764",
    events: "#6675b5",
    safety: "#a14949",
    subagent: "#7b62a0",
    ops: "#5d6a79",
  };

  function resolveRoot(config) {
    return new URL(config.rootPath || "./", window.location.href);
  }

  function resolveHref(config, href) {
    return new URL(href, resolveRoot(config)).href;
  }

  function isEmbedded() {
    return new URLSearchParams(window.location.search).get("embedded") === "1";
  }

  function currentPageId(config) {
    return config.pageId || "";
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function richList(items) {
    if (!items || !items.length) {
      return null;
    }

    const list = el("ul", "detail-list");
    items.forEach((item) => {
      const li = el("li");
      li.textContent = item;
      list.appendChild(li);
    });
    return list;
  }

  function annotationController(config, elements, getTarget) {
    const storageKey = `agent-server-visual:${config.pageId}:annotations`;
    let annotations = [];

    function load() {
      try {
        const raw = window.localStorage.getItem(storageKey);
        annotations = raw ? JSON.parse(raw) : [];
      } catch (_error) {
        annotations = [];
      }
    }

    function save() {
      window.localStorage.setItem(storageKey, JSON.stringify(annotations));
    }

    function markdown() {
      const lines = [`# ${config.title} annotations`, ""];
      if (!annotations.length) {
        lines.push("No annotations captured yet.");
        return lines.join("\n");
      }

      annotations.forEach((entry) => {
        lines.push(`## ${entry.target}`);
        lines.push(`- Type: ${entry.type}`);
        lines.push(`- Note: ${entry.text}`);
        if (entry.scenario) {
          lines.push(`- Scenario: ${entry.scenario}`);
        }
        lines.push("");
      });
      return lines.join("\n");
    }

    function render() {
      elements.target.textContent = `Target: ${getTarget().title}`;

      if (!annotations.length) {
        elements.list.innerHTML = "";
        const empty = el("div", "empty-copy", "No annotations yet. Capture open questions or review comments and export them as Markdown.");
        elements.list.appendChild(empty);
        return;
      }

      elements.list.innerHTML = "";
      annotations.forEach((entry) => {
        const item = el("div", "annotation-item");
        const top = el("div", "annotation-top");
        const badge = el("div", "annotation-badge", `${entry.type} · ${entry.target}`);
        const remove = el("button", "annotation-remove", "Remove");
        remove.addEventListener("click", () => {
          annotations = annotations.filter((candidate) => candidate.id !== entry.id);
          save();
          render();
        });
        top.appendChild(badge);
        top.appendChild(remove);
        item.appendChild(top);

        if (entry.scenario) {
          item.appendChild(el("div", "card-copy", `Scenario: ${entry.scenario}`));
        }

        item.appendChild(el("div", "", entry.text));
        elements.list.appendChild(item);
      });
    }

    function add() {
      const text = elements.input.value.trim();
      if (!text) {
        return;
      }

      const target = getTarget();
      annotations.push({
        id: `note-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        type: elements.type.value,
        text,
        target: target.title,
        scenario: target.scenario || null,
      });
      elements.input.value = "";
      save();
      render();
    }

    async function copy() {
      await navigator.clipboard.writeText(markdown());
    }

    function exportFile() {
      const blob = new Blob([markdown()], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${config.pageId}-annotations.md`;
      anchor.click();
      URL.revokeObjectURL(url);
    }

    load();
    render();
    elements.add.addEventListener("click", add);
    elements.copy.addEventListener("click", copy);
    elements.export.addEventListener("click", exportFile);

    return { render };
  }

  function buildSharedShell(config) {
    const app = document.getElementById("app");
    app.innerHTML = "";
    const shell = el("div", "app-shell");
    const topbar = el("header", "topbar");
    const titleWrap = el("div");
    titleWrap.appendChild(el("div", "eyebrow", config.kicker));
    titleWrap.appendChild(el("h1", "page-title", config.title));
    titleWrap.appendChild(el("p", "page-summary", config.summary));
    topbar.appendChild(titleWrap);

    const actions = el("div", "topbar-actions");
    if (config.kind !== "hub") {
      const hubLink = el("a", "pill-link strong", "Open Atlas Hub");
      hubLink.href = resolveHref(config, "architecture-atlas.html");
      actions.appendChild(hubLink);
    }
    PAGE_LIBRARY.flatMap((group) => group.items)
      .filter((page) => page.id === currentPageId(config))
      .forEach((page) => {
        const sourceLink = el("a", "pill-link", "Direct File");
        sourceLink.href = resolveHref(config, page.href);
        actions.appendChild(sourceLink);
      });
    topbar.appendChild(actions);
    shell.appendChild(topbar);
    app.appendChild(shell);
    return shell;
  }

  function buildPageNav(config) {
    const card = el("section", "sidebar-card");
    card.appendChild(el("div", "eyebrow", "Visual Set"));
    card.appendChild(el("h2", "card-heading", "Navigation"));
    card.appendChild(el("p", "card-copy", "Split views replace the dense board. Each page isolates one reasoning problem so arrows can stay orthogonal and readable."));

    PAGE_LIBRARY.forEach((group) => {
      const groupWrap = el("div", "nav-group");
      groupWrap.appendChild(el("div", "nav-group-label", group.title));
      const list = el("div", "nav-list");
      group.items.forEach((item) => {
        const link = el("a", `nav-item${item.id === currentPageId(config) ? " active" : ""}`);
        link.href = resolveHref(config, item.href);
        link.appendChild(el("strong", "", item.title));
        link.appendChild(el("span", "", item.copy));
        list.appendChild(link);
      });
      groupWrap.appendChild(list);
      card.appendChild(groupWrap);
    });
    return card;
  }

  function buildDetailCard() {
    const card = el("section", "detail-card");
    card.appendChild(el("div", "eyebrow", "Selected"));
    const body = el("div", "detail-body");
    const title = el("h3");
    const tags = el("div", "detail-tags");
    const copy = el("p");
    const bulletsHost = el("div");
    const questionsHost = el("div");
    body.appendChild(title);
    body.appendChild(tags);
    body.appendChild(copy);
    body.appendChild(bulletsHost);
    body.appendChild(questionsHost);
    card.appendChild(body);
    return {
      card,
      set(detail) {
        title.textContent = detail.title;
        tags.innerHTML = "";
        (detail.tags || []).forEach((tag) => tags.appendChild(el("div", "pill", tag)));
        copy.textContent = detail.copy;

        bulletsHost.innerHTML = "";
        const bullets = richList(detail.bullets);
        if (bullets) {
          bulletsHost.appendChild(bullets);
        }

        questionsHost.innerHTML = "";
        if (detail.questions && detail.questions.length) {
          questionsHost.appendChild(el("h4", "", "Open questions"));
          questionsHost.appendChild(richList(detail.questions));
        }
      },
    };
  }

  function buildAnnotationCard(config, getTarget) {
    const card = el("section", "annotation-card");
    card.appendChild(el("div", "eyebrow", "Review Notes"));
    const heading = el("h2", "card-heading", "Questions and comments");
    const copy = el("p", "card-copy", "Capture questions while reviewing and export them as Markdown to feed back into planning review.");
    const target = el("div", "pill");
    const type = el("select", "annotation-select");
    [["question", "Question"], ["note", "Note"], ["decision", "Decision"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      type.appendChild(option);
    });
    const input = el("textarea", "annotation-input");
    input.placeholder = "Add a question, missing assumption, or tradeoff note.";
    const actions = el("div", "annotation-actions");
    const add = el("button", "annotation-action strong", "Add");
    const copyBtn = el("button", "annotation-action", "Copy Markdown");
    const exportBtn = el("button", "annotation-action", "Export Markdown");
    actions.appendChild(add);
    actions.appendChild(copyBtn);
    actions.appendChild(exportBtn);
    const list = el("div", "annotation-list");

    card.appendChild(heading);
    card.appendChild(copy);
    card.appendChild(target);
    card.appendChild(type);
    card.appendChild(input);
    card.appendChild(actions);
    card.appendChild(list);

    const controller = annotationController(
      config,
      {
        target,
        type,
        input,
        add,
        copy: copyBtn,
        export: exportBtn,
        list,
      },
      getTarget,
    );

    return { card, refresh: controller.render };
  }

  function renderSequencePage(config) {
    const shell = buildSharedShell(config);
    const layout = el("div", "page-layout");
    const left = el("aside", "sidebar");
    const main = el("main", "main-card");
    const right = el("aside", "sidebar sidebar-right");

    left.appendChild(buildPageNav(config));
    const scenarioCard = el("section", "sidebar-card");
    scenarioCard.appendChild(el("div", "eyebrow", "Scenarios"));
    scenarioCard.appendChild(el("h2", "card-heading", "Timeline variants"));
    scenarioCard.appendChild(el("p", "card-copy", "Switch between success and failure paths without redrawing the whole system map."));
    const scenarioList = el("div", "scenario-list");
    scenarioCard.appendChild(scenarioList);
    left.appendChild(scenarioCard);

    const detail = buildDetailCard();
    right.appendChild(detail.card);

    let activeScenario = config.scenarios[0];
    let activeTarget = {
      title: activeScenario.title,
      scenario: activeScenario.title,
      detail: activeScenario.detail,
    };

    const annotation = buildAnnotationCard(config, () => activeTarget);
    right.appendChild(annotation.card);

    function setDetail(detailData, scenarioTitle) {
      activeTarget = {
        title: detailData.title,
        scenario: scenarioTitle,
        detail: detailData,
      };
      detail.set(detailData);
      annotation.refresh();
    }

    function renderBoard() {
      main.innerHTML = "";
      const header = el("div", "main-header");
      const headerText = el("div");
      headerText.appendChild(el("h2", "", activeScenario.title));
      headerText.appendChild(el("p", "", activeScenario.copy));
      header.appendChild(headerText);

      const legend = el("div", "legend");
      (activeScenario.legend || config.legend || []).forEach((item) => {
        const legendItem = el("div", "legend-item");
        const swatch = el("div", "legend-swatch");
        swatch.style.background = TONE_COLORS[item.tone] || item.color || "#999";
        legendItem.appendChild(swatch);
        legendItem.appendChild(el("span", "", item.label));
        legend.appendChild(legendItem);
      });
      header.appendChild(legend);
      main.appendChild(header);

      const frame = el("div", "main-frame");
      const board = el("div", "sequence-board");
      const laneHeader = el("div", "lane-header");
      const laneGrid = el("div", "lane-grid");
      laneGrid.style.gridTemplateColumns = `240px repeat(${config.lanes.length}, minmax(0, 1fr))`;
      laneGrid.appendChild(el("div"));
      config.lanes.forEach((lane) => {
        const laneCard = el("div", `lane-card ${lane.tone}`);
        laneCard.appendChild(el("div", "lane-kicker", lane.group || lane.tone));
        laneCard.appendChild(el("h3", "", lane.title));
        laneCard.appendChild(el("p", "", lane.copy));
        laneGrid.appendChild(laneCard);
      });
      laneHeader.appendChild(laneGrid);
      board.appendChild(laneHeader);

      activeScenario.steps.forEach((step, index) => {
        const row = el("div", "step-row");
        row.style.gridTemplateColumns = `240px repeat(${config.lanes.length}, minmax(0, 1fr))`;
        const stub = el("button", "step-stub");
        stub.appendChild(el("div", "step-count", `Step ${String(index + 1).padStart(2, "0")}`));
        stub.appendChild(el("strong", "", step.title));
        stub.appendChild(el("p", "", step.copy));
        stub.addEventListener("click", () => {
          board.querySelectorAll(".step-stub, .activity-card").forEach((node) => node.classList.remove("active"));
          stub.classList.add("active");
          setDetail(step.detail || {
            title: step.title,
            copy: step.copy,
            bullets: step.bullets,
            questions: step.questions,
            tags: step.tags,
          }, activeScenario.title);
        });
        row.appendChild(stub);

        step.cards.forEach((card) => {
          const laneIndex = config.lanes.findIndex((lane) => lane.id === card.lane);
          const button = el("button", `activity-card ${card.tone || "runtime"}`);
          button.style.gridColumn = `${laneIndex + 2} / span ${card.span || 1}`;
          if (card.offsetRows) {
            button.style.marginTop = `${card.offsetRows * 6}px`;
          }
          if (card.meta) {
            button.appendChild(el("div", "activity-meta", card.meta));
          }
          button.appendChild(el("h4", "", card.title));
          button.appendChild(el("p", "", card.copy));
          if (card.points && card.points.length) {
            const list = richList(card.points);
            if (list) {
              button.appendChild(list);
            }
          }
          button.addEventListener("click", () => {
            board.querySelectorAll(".step-stub, .activity-card").forEach((node) => node.classList.remove("active"));
            button.classList.add("active");
            setDetail(card.detail || {
              title: card.title,
              copy: card.copy,
              bullets: card.points || card.bullets,
              questions: card.questions,
              tags: card.tags || [card.meta].filter(Boolean),
            }, activeScenario.title);
          });
          row.appendChild(button);
        });

        board.appendChild(row);
      });

      frame.appendChild(board);
      main.appendChild(frame);
    }

    config.scenarios.forEach((scenario) => {
      const button = el("button", `scenario-button${scenario.id === activeScenario.id ? " active" : ""}`);
      button.appendChild(el("strong", "", scenario.title));
      button.appendChild(el("span", "", scenario.copy));
      button.addEventListener("click", () => {
        activeScenario = scenario;
        scenarioList.querySelectorAll(".scenario-button").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        setDetail(scenario.detail, scenario.title);
        renderBoard();
      });
      scenarioList.appendChild(button);
    });

    setDetail(activeScenario.detail, activeScenario.title);
    renderBoard();

    layout.appendChild(left);
    layout.appendChild(main);
    layout.appendChild(right);
    shell.appendChild(layout);
  }

  function renderStatePage(config) {
    const shell = buildSharedShell(config);
    const layout = el("div", "page-layout");
    const left = el("aside", "sidebar");
    const main = el("main", "main-card");
    const right = el("aside", "sidebar sidebar-right");

    left.appendChild(buildPageNav(config));
    const scenarioCard = el("section", "sidebar-card");
    scenarioCard.appendChild(el("div", "eyebrow", "Focus Modes"));
    scenarioCard.appendChild(el("h2", "card-heading", "Highlight one state story"));
    scenarioCard.appendChild(el("p", "card-copy", "These diagrams keep the same fixed layout. Focus modes only change the highlighted transitions and notes."));
    const scenarioList = el("div", "scenario-list");
    scenarioCard.appendChild(scenarioList);
    left.appendChild(scenarioCard);

    const detail = buildDetailCard();
    right.appendChild(detail.card);
    let activeScenario = config.scenarios[0];
    let activeTarget = { title: activeScenario.title, scenario: activeScenario.title };

    const annotation = buildAnnotationCard(config, () => activeTarget);
    right.appendChild(annotation.card);

    function setDetail(detailData, scenarioTitle) {
      activeTarget = {
        title: detailData.title,
        scenario: scenarioTitle,
        detail: detailData,
      };
      detail.set(detailData);
      annotation.refresh();
    }

    function renderBoard() {
      main.innerHTML = "";
      const header = el("div", "main-header");
      const headerText = el("div");
      headerText.appendChild(el("h2", "", activeScenario.title));
      headerText.appendChild(el("p", "", activeScenario.copy));
      header.appendChild(headerText);
      const legend = el("div", "legend");
      (config.legend || []).forEach((item) => {
        const legendItem = el("div", "legend-item");
        const swatch = el("div", "legend-swatch");
        swatch.style.background = TONE_COLORS[item.tone] || item.color || "#999";
        legendItem.appendChild(swatch);
        legendItem.appendChild(el("span", "", item.label));
        legend.appendChild(legendItem);
      });
      header.appendChild(legend);
      main.appendChild(header);

      const frame = el("div", "main-frame");
      const board = el("div", "state-board");
      board.style.width = `${config.boardWidth || 1200}px`;
      board.style.height = `${config.boardHeight || 760}px`;
      board.style.minHeight = `${config.boardHeight || 760}px`;
      const svgLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svgLayer.setAttribute("class", "edge-layer state-layer");
      svgLayer.setAttribute("viewBox", `0 0 ${config.boardWidth || 1200} ${config.boardHeight || 760}`);
      board.appendChild(svgLayer);
      const nodeLayer = el("div", "state-layer");
      board.appendChild(nodeLayer);

      const activeNodeIds = new Set(activeScenario.highlightNodes || []);
      const activeEdgeIds = new Set(activeScenario.highlightEdges || []);

      config.edges.forEach((edge) => {
        const pathValue = edge.points
          .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
          .join(" ");
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathValue);
        path.setAttribute("class", "edge-path");
        path.setAttribute("stroke", TONE_COLORS[edge.tone] || "#7f8a9b");
        path.style.opacity = activeEdgeIds.size ? (activeEdgeIds.has(edge.id) ? "1" : "0.22") : "0.82";
        svgLayer.appendChild(path);

        const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
        hit.setAttribute("d", pathValue);
        hit.setAttribute("class", "edge-hit");
        hit.addEventListener("click", () => {
          document.querySelectorAll(".state-node, .edge-label").forEach((node) => node.classList.remove("active"));
          label.classList.add("active");
          setDetail(edge.detail || {
            title: edge.label,
            copy: edge.copy,
            bullets: edge.pointsText,
            questions: edge.questions,
            tags: edge.tags || [edge.tone, "transition"],
          }, activeScenario.title);
        });
        svgLayer.appendChild(hit);

        const label = el("button", `edge-label${activeEdgeIds.has(edge.id) ? " active" : ""}`, edge.label);
        label.style.left = `${edge.labelX}px`;
        label.style.top = `${edge.labelY}px`;
        label.addEventListener("click", () => {
          document.querySelectorAll(".state-node, .edge-label").forEach((node) => node.classList.remove("active"));
          label.classList.add("active");
          setDetail(edge.detail || {
            title: edge.label,
            copy: edge.copy,
            bullets: edge.pointsText,
            questions: edge.questions,
            tags: edge.tags || [edge.tone, "transition"],
          }, activeScenario.title);
        });
        board.appendChild(label);
      });

      config.nodes.forEach((node) => {
        const button = el("button", `state-node ${node.tone}`);
        button.style.left = `${node.x}px`;
        button.style.top = `${node.y}px`;
        button.style.width = `${node.w}px`;
        button.style.minHeight = `${node.h}px`;
        button.style.opacity = activeNodeIds.size ? (activeNodeIds.has(node.id) ? "1" : "0.4") : "1";
        button.appendChild(el("div", "lane-kicker", node.group || node.tone));
        button.appendChild(el("h3", "", node.title));
        button.appendChild(el("p", "", node.copy));
        button.addEventListener("click", () => {
          document.querySelectorAll(".state-node, .edge-label").forEach((candidate) => candidate.classList.remove("active"));
          button.classList.add("active");
          setDetail(node.detail || {
            title: node.title,
            copy: node.copy,
            bullets: node.points,
            questions: node.questions,
            tags: node.tags || [node.group || node.tone],
          }, activeScenario.title);
        });
        nodeLayer.appendChild(button);
      });

      frame.appendChild(board);
      main.appendChild(frame);
    }

    config.scenarios.forEach((scenario) => {
      const button = el("button", `scenario-button${scenario.id === activeScenario.id ? " active" : ""}`);
      button.appendChild(el("strong", "", scenario.title));
      button.appendChild(el("span", "", scenario.copy));
      button.addEventListener("click", () => {
        activeScenario = scenario;
        scenarioList.querySelectorAll(".scenario-button").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        setDetail(scenario.detail, scenario.title);
        renderBoard();
      });
      scenarioList.appendChild(button);
    });

    setDetail(activeScenario.detail, activeScenario.title);
    renderBoard();

    layout.appendChild(left);
    layout.appendChild(main);
    layout.appendChild(right);
    shell.appendChild(layout);
  }

  function renderHubPage(config) {
    const shell = buildSharedShell(config);
    const layout = el("div", "hub-layout");
    const left = el("aside", "hub-sidebar");
    const right = el("main", "preview-card dashboard-main");

    const intro = el("section", "hub-card");
    intro.appendChild(el("div", "eyebrow", "Atlas Index"));
    intro.appendChild(el("h2", "card-heading", "Single dashboard navigation"));
    intro.appendChild(el("p", "card-copy", "Use this page as the primary review surface. Every flow and state view is loaded here without opening a new tab or window."));
    left.appendChild(intro);

    const flatPages = PAGE_LIBRARY.flatMap((group) => group.items);
    let activePage = flatPages[0];
    const pageLists = [];
    let focused = false;

    PAGE_LIBRARY.forEach((group) => {
      const card = el("section", "hub-card");
      card.appendChild(el("div", "eyebrow", group.title));
      card.appendChild(el("h2", "card-heading", group.title));
      const list = el("div", "page-list");
      group.items.forEach((item) => {
        const button = el("button", `page-chip${item.id === activePage.id ? " active" : ""}`);
        button.dataset.pageId = item.id;
        button.appendChild(el("strong", "", item.title));
        button.appendChild(el("span", "", item.copy));
        button.addEventListener("click", () => {
          activePage = item;
          pageLists.forEach((entry) => entry.querySelectorAll(".page-chip").forEach((node) => node.classList.remove("active")));
          button.classList.add("active");
          updatePreview();
        });
        list.appendChild(button);
      });
      pageLists.push(list);
      card.appendChild(list);
      left.appendChild(card);
    });

    const previewHeader = el("div", "dashboard-toolbar");
    const previewTitleWrap = el("div");
    const previewTitle = el("h2");
    const previewCopy = el("p");
    previewTitleWrap.appendChild(previewTitle);
    previewTitleWrap.appendChild(previewCopy);
    previewHeader.appendChild(previewTitleWrap);
    const previewActions = el("div", "dashboard-meta");
    const indexPill = el("div", "pill");
    const groupPill = el("div", "pill");
    const prevButton = el("button", "pill-link", "Previous");
    const nextButton = el("button", "pill-link", "Next");
    const focusButton = el("button", "pill-link strong", "Focus View");
    previewActions.appendChild(indexPill);
    previewActions.appendChild(groupPill);
    previewActions.appendChild(prevButton);
    previewActions.appendChild(nextButton);
    previewActions.appendChild(focusButton);
    previewHeader.appendChild(previewActions);
    right.appendChild(previewHeader);

    const frameShell = el("div", "dashboard-frame-shell");
    const previewFrame = document.createElement("iframe");
    previewFrame.className = "preview-frame dashboard-frame";
    previewFrame.loading = "lazy";
    frameShell.appendChild(previewFrame);

    const footer = el("div", "dashboard-footer");
    const footerLead = el("div", "", "Navigate left to switch views. The embedded page hides its own outer chrome so the diagram gets the screen.");
    const footerHint = el("div", "", "Recommended order: root turn, tool runtime, checkpoint recovery, event replay, root state, tool state, authority map, schema map.");
    footer.appendChild(footerLead);
    footer.appendChild(footerHint);
    frameShell.appendChild(footer);
    right.appendChild(frameShell);

    function setFocusState(nextFocused) {
      focused = nextFocused;
      document.body.classList.toggle("dashboard-focus", focused);
      focusButton.textContent = focused ? "Exit Focus" : "Focus View";
    }

    function updatePreview() {
      previewTitle.textContent = activePage.title;
      previewCopy.textContent = activePage.copy;
      const href = new URL(resolveHref(config, activePage.href));
      href.searchParams.set("embedded", "1");
      previewFrame.src = href.toString();
      const index = flatPages.findIndex((page) => page.id === activePage.id);
      indexPill.textContent = `View ${index + 1}/${flatPages.length}`;
      const owningGroup = PAGE_LIBRARY.find((group) => group.items.some((item) => item.id === activePage.id));
      groupPill.textContent = owningGroup ? owningGroup.title : "View";
      prevButton.disabled = index <= 0;
      nextButton.disabled = index >= flatPages.length - 1;
      prevButton.style.opacity = index <= 0 ? "0.5" : "1";
      nextButton.style.opacity = index >= flatPages.length - 1 ? "0.5" : "1";
    }

    prevButton.addEventListener("click", () => {
      const index = flatPages.findIndex((page) => page.id === activePage.id);
      if (index <= 0) {
        return;
      }
      activePage = flatPages[index - 1];
      pageLists.forEach((entry) => entry.querySelectorAll(".page-chip").forEach((node) => node.classList.remove("active")));
      pageLists.forEach((entry) => {
        const node = entry.querySelector(`.page-chip[data-page-id="${activePage.id}"]`);
        if (node) {
          node.classList.add("active");
        }
      });
      updatePreview();
    });

    nextButton.addEventListener("click", () => {
      const index = flatPages.findIndex((page) => page.id === activePage.id);
      if (index >= flatPages.length - 1) {
        return;
      }
      activePage = flatPages[index + 1];
      pageLists.forEach((entry) => entry.querySelectorAll(".page-chip").forEach((node) => node.classList.remove("active")));
      pageLists.forEach((entry) => {
        const node = entry.querySelector(`.page-chip[data-page-id="${activePage.id}"]`);
        if (node) {
          node.classList.add("active");
        }
      });
      updatePreview();
    });

    focusButton.addEventListener("click", () => setFocusState(!focused));

    updatePreview();

    layout.appendChild(left);
    layout.appendChild(right);
    shell.appendChild(layout);
  }

  function renderPage() {
    const config = window.AGENT_ATLAS_PAGE;
    if (!config) {
      return;
    }
    document.body.classList.toggle("embedded-mode", isEmbedded());
    if (config.kind === "sequence") {
      renderSequencePage(config);
    } else if (config.kind === "state") {
      renderStatePage(config);
    } else if (config.kind === "hub") {
      renderHubPage(config);
    }
  }

  document.addEventListener("DOMContentLoaded", renderPage);
})();
