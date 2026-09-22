/**
 * परिधि (Paridhi) — Interactive Cadastral Workspace Client Engine
 * Features:
 * - Dynamic Layer Classification Filter (Show / Hide individual classes)
 * - 5-Option Land Use Dropdown (Vacant, Residential, Commercial, Agriculture, Mixed Use)
 * - Parcel Inspection & Verification Modal with 2 verification options (Verified / Disputed)
 * - Canvas Pan / Zoom with high-precision vector rendering
 */

class CadastralWorkspace {
  constructor() {
    this.canvas = document.getElementById("gisCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.container = document.getElementById("canvasContainer");

    // State
    this.analysisData = null;
    this.currentMode = "overlay"; // 'overlay', 'wireframe', 'original'
    this.selectedParcelId = null;
    this.images = {
      original: null,
      overlay: null,
      wireframe: null
    };

    // Layer Visibility Filter
    this.visibleClasses = new Set([
      "Building",
      "Tree / Forest",
      "Agricultural Land",
      "Playground / Open Turf",
      "Road / Pathway",
      "Vacant / Open Land"
    ]);

    // Viewport transform (Pan & Zoom)
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    this.initElements();
    this.initEventListeners();
    this.resizeCanvas();
    this.loadInitialData();
  }

  initElements() {
    this.elImageSelect = document.getElementById("imageSelect");
    this.elBtnRun = document.getElementById("btnRunAnalysis");
    this.elBtnRunText = document.getElementById("btnRunText");
    this.elLoader = document.getElementById("loaderOverlay");
    this.elTimerBadge = document.getElementById("analysisTimerBadge");
    this.elTimerVal = document.getElementById("timerVal");

    // KPIs
    this.elValBldgs = document.getElementById("valBldgs");
    this.elValTrees = document.getElementById("valTrees");
    this.elValAgri = document.getElementById("valAgri");
    this.elValTurf = document.getElementById("valTurf");
    this.elValArea = document.getElementById("valArea");
    this.elValAreaSub = document.getElementById("valAreaSub");
    this.elValDensity = document.getElementById("valDensity");

    // Legend Counts
    this.elCountBldg = document.getElementById("countBldg");
    this.elCountTree = document.getElementById("countTree");
    this.elCountAgri = document.getElementById("countAgri");
    this.elCountTurf = document.getElementById("countTurf");
    this.elCountRoad = document.getElementById("countRoad");
    this.elCountVacant = document.getElementById("countVacant");

    // Inputs
    this.elInputGsd = document.getElementById("inputGsd");
    this.elInputAlpha = document.getElementById("inputAlpha");
    this.elAlphaVal = document.getElementById("alphaVal");

    // Table
    this.elTableBody = document.getElementById("tableBody");
    this.elTableSearch = document.getElementById("tableSearch");

    // Exports
    this.elBtnExportGeoJSON = document.getElementById("btnExportGeoJSON");
    this.elBtnExportCSV = document.getElementById("btnExportCSV");
    this.elBtnExportOverlay = document.getElementById("btnExportOverlay");

    // Modal Elements
    this.elModal = document.getElementById("parcelModal");
    this.elModalIdBadge = document.getElementById("modalParcelIdBadge");
    this.elModalIdNo = document.getElementById("modalIdNo");
    this.elModalFeatureDot = document.getElementById("modalFeatureDot");
    this.elModalFeatureText = document.getElementById("modalFeatureText");
    this.elModalLandUseSelect = document.getElementById("modalLandUseSelect");
    this.elModalAreaSqm = document.getElementById("modalAreaSqm");
    this.elModalAreaGaj = document.getElementById("modalAreaGaj");
    this.elModalPerimeter = document.getElementById("modalPerimeter");
    this.elModalConfidence = document.getElementById("modalConfidence");
    this.elModalVertexCount = document.getElementById("modalVertexCount");
    this.elModalStatusBadge = document.getElementById("modalStatusBadge");
    this.elModalBtnVerify = document.getElementById("modalBtnMarkVerified");
    this.elModalBtnDispute = document.getElementById("modalBtnMarkDisputed");
    this.elBtnCloseModal = document.getElementById("btnCloseModal");
    this.elBtnModalDone = document.getElementById("btnModalDone");
  }

  initEventListeners() {
    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.render();
    });

    if (this.elBtnRun) {
      this.elBtnRun.addEventListener("click", () => this.runAnalysis());
    }

    if (this.elImageSelect) {
      this.elImageSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        this.loadOriginalImage(val, () => this.runAnalysis());
      });
    }

    if (this.elInputAlpha && this.elAlphaVal) {
      this.elInputAlpha.addEventListener("input", (e) => {
        this.elAlphaVal.textContent = `${Math.round(e.target.value * 100)}%`;
        this.render();
      });
    }

    // View Mode Tabs
    document.querySelectorAll(".view-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        this.currentMode = tab.getAttribute("data-mode");
        this.render();
      });
    });

    // Layer Classification Checkbox Filters
    document.querySelectorAll(".class-filter-checkbox").forEach(cb => {
      cb.addEventListener("change", (e) => {
        const cname = e.target.getAttribute("data-class");
        if (e.target.checked) {
          this.visibleClasses.add(cname);
        } else {
          this.visibleClasses.delete(cname);
        }
        this.render();
        this.populateTable();
      });
    });

    // Select All / Clear All Classes
    document.getElementById("btnSelectAllClasses")?.addEventListener("click", () => {
      document.querySelectorAll(".class-filter-checkbox").forEach(cb => {
        cb.checked = true;
        this.visibleClasses.add(cb.getAttribute("data-class"));
      });
      this.render();
      this.populateTable();
    });

    document.getElementById("btnClearAllClasses")?.addEventListener("click", () => {
      document.querySelectorAll(".class-filter-checkbox").forEach(cb => {
        cb.checked = false;
        this.visibleClasses.delete(cb.getAttribute("data-class"));
      });
      this.render();
      this.populateTable();
    });

    // Zoom Controls
    document.getElementById("btnZoomIn")?.addEventListener("click", () => this.zoomAtCenter(1.25));
    document.getElementById("btnZoomOut")?.addEventListener("click", () => this.zoomAtCenter(0.8));
    document.getElementById("btnResetView")?.addEventListener("click", () => this.resetView());

    // Canvas Pan & Drag
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
      this.canvas.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      this.panX = e.clientX - this.dragStartX;
      this.panY = e.clientY - this.dragStartY;
      this.render();
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
      this.canvas.style.cursor = "grab";
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.max(0.1, Math.min(this.zoom * factor, 15.0));

      this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
      this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
      this.zoom = newZoom;
      this.render();
    }, { passive: false });

    // Table Search
    this.elTableSearch?.addEventListener("input", () => this.populateTable());

    // Export Buttons
    this.elBtnExportGeoJSON?.addEventListener("click", () => this.exportFile("geojson"));
    this.elBtnExportCSV?.addEventListener("click", () => this.exportFile("csv"));
    this.elBtnExportOverlay?.addEventListener("click", () => this.exportFile("overlay"));

    // Modal Close
    this.elBtnCloseModal?.addEventListener("click", () => this.closeModal());
    this.elBtnModalDone?.addEventListener("click", () => this.closeModal());
    this.elModal?.addEventListener("click", (e) => {
      if (e.target === this.elModal) this.closeModal();
    });

    // Modal Actions
    this.elModalBtnVerify?.addEventListener("click", () => {
      if (this.selectedParcelId) {
        this.updateVerification(this.selectedParcelId, "Verified");
      }
    });

    this.elModalBtnDispute?.addEventListener("click", () => {
      if (this.selectedParcelId) {
        this.updateVerification(this.selectedParcelId, "Disputed");
      }
    });

    this.elModalLandUseSelect?.addEventListener("change", (e) => {
      if (this.selectedParcelId) {
        this.updateLandUse(this.selectedParcelId, e.target.value);
      }
    });
  }

  resizeCanvas() {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);
  }

  async loadInitialData() {
    try {
      const resp = await fetch("/api/sample-images");
      const data = await resp.json();

      this.elImageSelect.innerHTML = "";

      const uploadedFilename = localStorage.getItem("uploadedImageFilename");
      const uploadedUrl = localStorage.getItem("uploadedImageUrl");

      if (uploadedFilename && uploadedUrl) {
        const upOpt = new Option(`⭐ Recently Uploaded: ${uploadedFilename}`, uploadedUrl, true, true);
        this.elImageSelect.add(upOpt);
      }

      if (data.samples && data.samples.length > 0) {
        data.samples.forEach(s => {
          const opt = new Option(`Sample: ${s.filename}`, s.url);
          this.elImageSelect.add(opt);
        });
      }

      const activeUrl = this.elImageSelect.value;
      if (activeUrl) {
        this.loadOriginalImage(activeUrl, () => this.runAnalysis());
      }
    } catch (err) {
      console.warn("Could not fetch samples, using fallback:", err);
    }
  }

  loadOriginalImage(url, callback) {
    this.images.original = new Image();
    this.images.original.onload = () => {
      this.resetView();
      if (callback) callback();
    };
    this.images.original.onerror = () => {
      console.error("Failed to load original image:", url);
      if (callback) callback();
    };
    this.images.original.src = url;
  }

  resetView() {
    if (!this.images.original || !this.images.original.width) return;
    const rect = this.container.getBoundingClientRect();
    const scaleX = rect.width / this.images.original.width;
    const scaleY = rect.height / this.images.original.height;
    this.zoom = Math.min(scaleX, scaleY) * 0.94;
    this.panX = (rect.width - this.images.original.width * this.zoom) / 2;
    this.panY = (rect.height - this.images.original.height * this.zoom) / 2;
    this.render();
  }

  zoomAtCenter(factor) {
    const rect = this.container.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const newZoom = Math.max(0.1, Math.min(this.zoom * factor, 15.0));

    this.panX = cx - (cx - this.panX) * (newZoom / this.zoom);
    this.panY = cy - (cy - this.panY) * (newZoom / this.zoom);
    this.zoom = newZoom;
    this.render();
  }

  async runAnalysis() {
    const activeUrl = this.elImageSelect.value;
    if (!activeUrl) return;

    this.setLoading(true);
    const t0 = performance.now();

    const payload = {
      image_name: activeUrl.split("/").pop(),
      image_path: activeUrl,
      gsd_meters: parseFloat(this.elInputGsd.value) || 0.08
    };

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        throw new Error(`Analysis error (${resp.status}): ${await resp.text()}`);
      }

      const data = await resp.json();
      this.analysisData = data;

      const duration = ((performance.now() - t0) / 1000).toFixed(2);
      this.elTimerBadge.style.display = "inline-flex";
      this.elTimerVal.textContent = `${duration}s`;

      this.updateKPIs();
      this.populateTable();
      this.render();

    } catch (err) {
      console.error("Pipeline failure:", err);
      alert("Failed to analyze image: " + err.message);
    } finally {
      this.setLoading(false);
    }
  }

  setLoading(isLoading) {
    if (this.elLoader) this.elLoader.style.display = isLoading ? "flex" : "none";
    if (this.elBtnRun) {
      this.elBtnRun.disabled = isLoading;
      this.elBtnRunText.textContent = isLoading ? "Processing YOLOv8..." : "Run AI Analysis";
    }
  }

  updateKPIs() {
    if (!this.analysisData || !this.analysisData.summary) return;
    const s = this.analysisData.summary;

    this.elValBldgs.textContent = s.total_buildings || 0;
    this.elValTrees.textContent = s.total_trees || 0;
    this.elValAgri.textContent = s.total_farmlands || 0;
    this.elValTurf.textContent = s.total_playgrounds || 0;
    this.elValArea.textContent = `${(s.total_surveyed_area_sqm || 0).toLocaleString()} m²`;
    this.elValAreaSub.textContent = `${s.total_surveyed_hectares || 0} ha | ${s.total_surveyed_acres || 0} ac | ${(s.total_surveyed_gaj || 0).toLocaleString()} gaj`;
    this.elValDensity.textContent = `${s.builtup_pct || 0}%`;

    // Update legend count badges
    if (this.elCountBldg) this.elCountBldg.textContent = s.total_buildings || 0;
    if (this.elCountTree) this.elCountTree.textContent = s.total_trees || 0;
    if (this.elCountAgri) this.elCountAgri.textContent = s.total_farmlands || 0;
    if (this.elCountTurf) this.elCountTurf.textContent = s.total_playgrounds || 0;
    if (this.elCountRoad) this.elCountRoad.textContent = s.total_roads || 0;
    if (this.elCountVacant) this.elCountVacant.textContent = s.total_vacant || 0;
  }

  populateTable() {
    if (!this.analysisData || !this.analysisData.parcels) return;
    const q = (this.elTableSearch.value || "").trim().toLowerCase();

    const filtered = this.analysisData.parcels.filter(p => {
      const isClassVisible = this.visibleClasses.has(p.feature_type);
      if (!isClassVisible) return false;

      const currentLandUse = p.land_use || "Vacant";
      return !q ||
        p.parcel_id.toLowerCase().includes(q) ||
        p.feature_type.toLowerCase().includes(q) ||
        currentLandUse.toLowerCase().includes(q);
    });

    if (filtered.length === 0) {
      this.elTableBody.innerHTML = `<tr><td colspan="9" class="empty-msg">No matching cadastral features.</td></tr>`;
      return;
    }

    const landUseOptions = ["Vacant", "Residential", "Commercial", "Agriculture", "Mixed Use"];

    let rowsHtml = "";
    filtered.forEach(p => {
      const currentLandUse = p.land_use || "Vacant";
      const isSelected = p.parcel_id === this.selectedParcelId;

      const statusBadge = p.verification_status === "Verified"
        ? `<span class="badge verified">Verified ✅</span>`
        : (p.verification_status === "Disputed"
          ? `<span class="badge disputed">Disputed ⚠️</span>`
          : `<span class="badge pending">Pending ⏳</span>`);

      // 5-option select dropdown
      const selectHtml = `
        <select class="table-select landuse-select" data-id="${p.parcel_id}">
          ${landUseOptions.map(opt => `<option value="${opt}" ${opt === currentLandUse ? 'selected' : ''}>${opt}</option>`).join("")}
        </select>
      `;

      rowsHtml += `
        <tr class="parcel-row ${isSelected ? 'row-selected' : ''}" data-id="${p.parcel_id}">
          <td><strong>${p.parcel_id}</strong></td>
          <td>
            <span class="color-dot" style="background-color: ${p.color_hex};"></span>
            ${p.feature_type}
          </td>
          <td>${selectHtml}</td>
          <td>${(p.area_sqm || 0).toLocaleString()}</td>
          <td>${(p.area_gaj || 0).toLocaleString()}</td>
          <td>${p.perimeter_m || 0} m</td>
          <td><span class="badge conf">${Math.round((p.confidence || 0.85) * 100)}%</span></td>
          <td>${statusBadge}</td>
          <td>
            <button class="btn-action-inspect" data-id="${p.parcel_id}" title="Inspect Details & Verify">
              <i class="fa-solid fa-eye"></i> Details
            </button>
          </td>
        </tr>
      `;
    });

    this.elTableBody.innerHTML = rowsHtml;

    // Land-Use dropdown change listener
    this.elTableBody.querySelectorAll(".landuse-select").forEach(sel => {
      sel.addEventListener("change", (e) => {
        const pid = sel.getAttribute("data-id");
        this.updateLandUse(pid, e.target.value);
      });
    });

    // Inspect Details button click listener
    this.elTableBody.querySelectorAll(".btn-action-inspect").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const pid = btn.getAttribute("data-id");
        this.openModal(pid);
      });
    });

    // Row click to highlight & center
    this.elTableBody.querySelectorAll(".parcel-row").forEach(row => {
      row.addEventListener("click", (e) => {
        if (e.target.tagName === "SELECT" || e.target.tagName === "BUTTON" || e.target.closest("button")) return;
        const pid = row.getAttribute("data-id");
        this.selectParcel(pid);
      });
    });
  }

  selectParcel(parcelId) {
    this.selectedParcelId = parcelId;
    this.populateTable();
    this.centerOnParcel(parcelId);
    this.render();
  }

  centerOnParcel(parcelId) {
    if (!this.analysisData || !this.analysisData.geojson_data) return;
    const feat = this.analysisData.geojson_data.features.find(f => f.id === parcelId);
    if (!feat || !feat.geometry || !feat.geometry.coordinates[0]) return;

    const ring = feat.geometry.coordinates[0];
    let sumX = 0, sumY = 0;
    ring.forEach(pt => { sumX += pt[0]; sumY += pt[1]; });
    const cx = sumX / ring.length;
    const cy = sumY / ring.length;

    const rect = this.container.getBoundingClientRect();
    this.panX = rect.width / 2 - cx * this.zoom;
    this.panY = rect.height / 2 - cy * this.zoom;
    this.render();
  }

  openModal(parcelId) {
    this.selectedParcelId = parcelId;
    const p = this.analysisData?.parcels?.find(x => x.parcel_id === parcelId);
    if (!p) return;

    this.elModalIdBadge.textContent = p.parcel_id;
    this.elModalIdNo.textContent = p.parcel_id;
    this.elModalFeatureText.textContent = p.feature_type;
    this.elModalFeatureDot.style.background = p.color_hex;

    this.elModalLandUseSelect.value = p.land_use || "Vacant";
    this.elModalAreaSqm.textContent = `${(p.area_sqm || 0).toLocaleString()} m²`;
    this.elModalAreaGaj.textContent = `(${ (p.area_gaj || 0).toLocaleString() } Gaj)`;
    this.elModalPerimeter.textContent = `${p.perimeter_m || 0} meters`;
    this.elModalConfidence.textContent = `${Math.round((p.confidence || 0.85) * 100)}%`;
    this.elModalVertexCount.textContent = `${p.vertex_count || 0} coordinate points`;

    this.updateModalStatusBadge(p.verification_status);

    this.elModal.style.display = "flex";
    this.centerOnParcel(parcelId);
  }

  updateModalStatusBadge(status) {
    this.elModalStatusBadge.className = "badge";
    if (status === "Verified") {
      this.elModalStatusBadge.classList.add("verified");
      this.elModalStatusBadge.textContent = "Verified ✅";
    } else if (status === "Disputed") {
      this.elModalStatusBadge.classList.add("disputed");
      this.elModalStatusBadge.textContent = "Disputed ⚠️";
    } else {
      this.elModalStatusBadge.classList.add("pending");
      this.elModalStatusBadge.textContent = "Pending Review ⏳";
    }
  }

  closeModal() {
    if (this.elModal) this.elModal.style.display = "none";
    this.render();
  }

  async updateLandUse(parcelId, newLandUse) {
    const p = this.analysisData?.parcels?.find(x => x.parcel_id === parcelId);
    if (p) p.land_use = newLandUse;

    if (this.elModal && this.elModal.style.display !== "none" && this.selectedParcelId === parcelId) {
      this.elModalLandUseSelect.value = newLandUse;
    }

    try {
      await fetch("/api/parcels/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: this.analysisData.summary.analysis_id,
          parcel_id: parcelId,
          land_use: newLandUse
        })
      });
    } catch (e) {
      console.error("Land use update error:", e);
    }
    this.populateTable();
  }

  async updateVerification(parcelId, newStatus) {
    const p = this.analysisData?.parcels?.find(x => x.parcel_id === parcelId);
    if (p) p.verification_status = newStatus;

    this.updateModalStatusBadge(newStatus);

    try {
      await fetch("/api/parcels/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: this.analysisData.summary.analysis_id,
          parcel_id: parcelId,
          verification_status: newStatus
        })
      });
    } catch (e) {
      console.error("Verification update error:", e);
    }
    this.populateTable();
  }

  hexToRgba(hex, alpha) {
    let c = hex.replace("#", "");
    if (c.length === 3) c = c.split("").map(x => x + x).join("");
    const num = parseInt(c, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  render() {
    const rect = this.container.getBoundingClientRect();
    this.ctx.clearRect(0, 0, rect.width, rect.height);

    const baseImg = this.images.original;
    if (!baseImg || !baseImg.complete) return;

    this.ctx.save();
    this.ctx.translate(this.panX, this.panY);
    this.ctx.scale(this.zoom, this.zoom);

    // 1. Base Layer
    if (this.currentMode === "wireframe") {
      this.ctx.fillStyle = "#0a0f1d";
      this.ctx.fillRect(0, 0, baseImg.width, baseImg.height);
    } else {
      this.ctx.drawImage(baseImg, 0, 0);
    }

    // 2. Vector Layer Rendering with Selective Layer Filtering
    if (this.currentMode !== "original" && this.analysisData && this.analysisData.geojson_data) {
      const alpha = parseFloat(this.elInputAlpha?.value) || 0.42;
      const features = this.analysisData.geojson_data.features;

      // Draw order: Land first, then Trees, Roads, Buildings on top
      const order = { "Vacant / Open Land": 1, "Agricultural Land": 2, "Playground / Open Turf": 3, "Road / Pathway": 4, "Tree / Forest": 5, "Building": 6 };
      const sorted = [...features].sort((a, b) => {
        return (order[a.properties.feature_type] || 0) - (order[b.properties.feature_type] || 0);
      });

      // Pass 1: Semi-transparent Fills
      sorted.forEach(f => {
        const cname = f.properties.feature_type;
        if (!this.visibleClasses.has(cname)) return;

        const ring = f.geometry.coordinates[0];
        if (!ring || ring.length < 3) return;

        this.ctx.beginPath();
        this.ctx.moveTo(ring[0][0], ring[0][1]);
        for (let i = 1; i < ring.length; i++) {
          this.ctx.lineTo(ring[i][0], ring[i][1]);
        }
        this.ctx.closePath();

        const isWireframe = this.currentMode === "wireframe";
        this.ctx.fillStyle = this.hexToRgba(f.properties.color_hex, isWireframe ? 0.22 : alpha);
        this.ctx.fill();
      });

      // Pass 2: Vector Outlines & Highlight
      sorted.forEach(f => {
        const cname = f.properties.feature_type;
        if (!this.visibleClasses.has(cname)) return;

        const ring = f.geometry.coordinates[0];
        if (!ring || ring.length < 3) return;

        const isSelected = f.id === this.selectedParcelId;

        this.ctx.beginPath();
        this.ctx.moveTo(ring[0][0], ring[0][1]);
        for (let i = 1; i < ring.length; i++) {
          this.ctx.lineTo(ring[i][0], ring[i][1]);
        }
        this.ctx.closePath();

        if (isSelected) {
          this.ctx.strokeStyle = "#00ffff";
          this.ctx.lineWidth = Math.max(3, 4 / this.zoom);
          this.ctx.shadowColor = "#00ffff";
          this.ctx.shadowBlur = 10;
        } else {
          this.ctx.strokeStyle = f.properties.color_hex;
          this.ctx.lineWidth = Math.max(1.5, 2 / this.zoom);
          this.ctx.shadowBlur = 0;
        }

        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
      });
    }

    this.ctx.restore();
  }

  exportFile(format) {
    if (!this.analysisData || !this.analysisData.summary) {
      alert("Please run an AI analysis first before exporting data.");
      return;
    }
    const id = this.analysisData.summary.analysis_id;
    window.open(`/api/export/${id}/${format}`, "_blank");
  }
}

// Bootstrap
window.addEventListener("DOMContentLoaded", () => {
  window.cadastralApp = new CadastralWorkspace();
});
