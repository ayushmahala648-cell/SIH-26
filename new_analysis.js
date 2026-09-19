/**
 * परिधि (Paridhi) — Interactive Cadastral GIS Dashboard Engine
 * Handles high-performance Canvas pan/zoom, vector rendering,
 * spatial inspection, human verification, and REST API communication.
 */

class CadastralApp {
  constructor() {
    this.canvas = document.getElementById("gisCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.container = document.getElementById("canvasContainer");
    this.tooltip = document.getElementById("canvasTooltip");

    // State
    this.analysisData = null;
    this.baseImage = null;
    this.overlayImage = null;
    this.showOverlayImage = false;
    this.selectedParcelId = null;
    this.hoveredParcel = null;

    // Viewport Transform (Pan & Zoom)
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    // Layer Visibility
    this.layers = {
      ortho: true,
      parcels: true,
      buildings: true,
      roads: true,
      labels: true,
      orthoOpacity: 1.0,
      parcelOpacity: 0.75
    };

    this.initElements();
    this.initEventListeners();
    this.resizeCanvas();

    // Check URL parameters or sessionStorage
    this.bootstrapFromState();
  }

  initElements() {
    this.elImageSelector = document.getElementById("imageSelector");
    this.elBtnRun = document.getElementById("btnRunAnalysis");
    this.elBtnRunText = document.getElementById("btnRunText");
    this.elTimer = document.getElementById("analysisTimer");
    this.elTimerVal = document.getElementById("timerVal");
    this.elProcessingOverlay = document.getElementById("processingOverlay");

    this.elGsdInput = document.getElementById("gsdInput");
    this.elSimplifyTolInput = document.getElementById("simplifyTolInput");
    this.elSimplifyVal = document.getElementById("simplifyVal");
    this.elMinAreaInput = document.getElementById("minAreaInput");
    this.elSnapOrthogonal = document.getElementById("snapOrthogonalCheck");
    this.elBtnApplyParams = document.getElementById("btnApplyParams");

    this.elValParcels = document.getElementById("valTotalParcels");
    this.elValBuildings = document.getElementById("valTotalBuildings");
    this.elValArea = document.getElementById("valTotalArea");
    this.elValBuiltup = document.getElementById("valBuiltupPct");
    this.elBadgeParcelCount = document.getElementById("badgeParcelCount");

    this.elTableBody = document.getElementById("cadastralTableBody");
    this.elParcelSearch = document.getElementById("parcelSearch");
    this.elStatusFilter = document.getElementById("statusFilter");

    this.elBtnExportGeoJSON = document.getElementById("btnExportGeoJSON");
    this.elBtnExportCSV = document.getElementById("btnExportCSV");
    this.elBtnExportOverlay = document.getElementById("btnExportOverlay");

    this.elInspector = document.getElementById("parcelInspector");
    this.elInspectorBody = document.getElementById("inspectorBody");
    this.elInspParcelId = document.getElementById("inspParcelId");
    this.elCloseInspector = document.getElementById("closeInspectorBtn");
  }

  initEventListeners() {
    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.render();
    });

    // Simplify tolerance slider display
    if (this.elSimplifyTolInput && this.elSimplifyVal) {
      this.elSimplifyTolInput.addEventListener("input", (e) => {
        this.elSimplifyVal.textContent = `${e.target.value}px`;
      });
    }

    // Run / Re-analyze
    if (this.elBtnRun) {
      this.elBtnRun.addEventListener("click", () => this.runAnalysis());
    }
    if (this.elBtnApplyParams) {
      this.elBtnApplyParams.addEventListener("click", () => this.runAnalysis());
    }

    // Image Selector switch
    if (this.elImageSelector) {
      this.elImageSelector.addEventListener("change", (e) => {
        this.loadImage(e.target.value, () => this.runAnalysis());
      });
    }

    // Layer Controls
    document.getElementById("layerOrtho")?.addEventListener("change", (e) => {
      this.layers.ortho = e.target.checked;
      this.render();
    });
    document.getElementById("layerParcels")?.addEventListener("change", (e) => {
      this.layers.parcels = e.target.checked;
      this.render();
    });
    document.getElementById("layerBuildings")?.addEventListener("change", (e) => {
      this.layers.buildings = e.target.checked;
      this.render();
    });
    document.getElementById("layerRoads")?.addEventListener("change", (e) => {
      this.layers.roads = e.target.checked;
      this.render();
    });
    document.getElementById("layerLabels")?.addEventListener("change", (e) => {
      this.layers.labels = e.target.checked;
      this.render();
    });
    document.getElementById("orthoOpacity")?.addEventListener("input", (e) => {
      this.layers.orthoOpacity = parseFloat(e.target.value);
      this.render();
    });
    document.getElementById("parcelOpacity")?.addEventListener("input", (e) => {
      this.layers.parcelOpacity = parseFloat(e.target.value);
      this.render();
    });

    // Viewport Controls
    document.getElementById("btnZoomIn")?.addEventListener("click", () => this.zoomAtCenter(1.25));
    document.getElementById("btnZoomOut")?.addEventListener("click", () => this.zoomAtCenter(0.8));
    document.getElementById("btnResetZoom")?.addEventListener("click", () => this.resetView());
    document.getElementById("btnToggleOverlay")?.addEventListener("click", () => {
      this.showOverlayImage = !this.showOverlayImage;
      const btn = document.getElementById("btnToggleOverlay");
      if (btn) btn.classList.toggle("active", this.showOverlayImage);
      this.render();
    });

    // Canvas Mouse Handlers
    this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    this.canvas.addEventListener("mouseup", (e) => this.onMouseUp(e));
    this.canvas.addEventListener("mouseleave", () => this.onMouseLeave());
    this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener("click", (e) => this.onCanvasClick(e));

    // Search and Filter
    this.elParcelSearch?.addEventListener("input", () => this.populateTable());
    this.elStatusFilter?.addEventListener("change", () => this.populateTable());

    // Exports
    this.elBtnExportGeoJSON?.addEventListener("click", () => this.exportFile("geojson"));
    this.elBtnExportCSV?.addEventListener("click", () => this.exportFile("csv"));
    this.elBtnExportOverlay?.addEventListener("click", () => this.exportFile("overlay"));

    // Inspector Close
    this.elCloseInspector?.addEventListener("click", () => {
      this.elInspector.classList.remove("open");
    });
  }

  bootstrapFromState() {
    const urlParams = new URLSearchParams(window.location.search);
    const sampleParam = urlParams.get("sample") || urlParams.get("name") || sessionStorage.getItem("paridhi_image_name");
    const typeParam = urlParams.get("type") || sessionStorage.getItem("paridhi_image_type");
    const pathParam = urlParams.get("path") || sessionStorage.getItem("paridhi_image_path");

    if (sampleParam && this.elImageSelector) {
      // Find matching option in dropdown or append
      let found = false;
      for (let opt of this.elImageSelector.options) {
        if (opt.value === sampleParam) {
          this.elImageSelector.value = sampleParam;
          found = true;
          break;
        }
      }
      if (!found && typeParam === "uploaded") {
        const newOpt = new Option(`Uploaded: ${sampleParam}`, pathParam, true, true);
        this.elImageSelector.add(newOpt);
      }
    }

    const currentImg = this.elImageSelector.value;
    this.loadImage(currentImg, () => {
      this.runAnalysis();
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

  loadImage(imageSource, callback) {
    let src = imageSource;
    if (!src.startsWith("/") && !src.startsWith("http") && !src.startsWith("uploads/")) {
      src = `/images/${imageSource}`;
    }

    this.baseImage = new Image();
    this.baseImage.onload = () => {
      this.resetView();
      if (callback) callback();
    };
    this.baseImage.onerror = () => {
      console.warn("Could not load image source:", src);
      if (callback) callback();
    };
    this.baseImage.src = src;
  }

  resetView() {
    if (!this.baseImage || !this.baseImage.width) return;
    const rect = this.container.getBoundingClientRect();
    const scaleX = rect.width / this.baseImage.width;
    const scaleY = rect.height / this.baseImage.height;
    this.zoom = Math.min(scaleX, scaleY) * 0.95;
    this.panX = (rect.width - this.baseImage.width * this.zoom) / 2;
    this.panY = (rect.height - this.baseImage.height * this.zoom) / 2;
    this.render();
  }

  zoomAtCenter(factor) {
    const rect = this.container.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    this.zoomAtPoint(centerX, centerY, factor);
  }

  zoomAtPoint(screenX, screenY, factor) {
    const newZoom = Math.max(0.05, Math.min(this.zoom * factor, 20.0));
    const ratio = newZoom / this.zoom;
    this.panX = screenX - (screenX - this.panX) * ratio;
    this.panY = screenY - (screenY - this.panY) * ratio;
    this.zoom = newZoom;
    this.render();
  }

  // API Call
  async runAnalysis() {
    this.setLoading(true);
    const t0 = performance.now();

    const selectedVal = this.elImageSelector.value;
    const payload = {
      gsd_meters: parseFloat(this.elGsdInput.value) || 0.08,
      simplify_tolerance: parseFloat(this.elSimplifyTolInput.value) || 2.0,
      min_parcel_area_px: parseInt(this.elMinAreaInput.value, 10) || 100,
      snap_orthogonal: this.elSnapOrthogonal.checked
    };

    if (selectedVal.includes("/") || selectedVal.includes("\\")) {
      payload.image_path = selectedVal;
    } else {
      payload.image_name = selectedVal;
    }

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      this.analysisData = data;

      // Load overlay image
      if (data.overlay_url) {
        this.overlayImage = new Image();
        this.overlayImage.src = data.overlay_url;
      }

      const duration = ((performance.now() - t0) / 1000).toFixed(2);
      this.elTimer.style.display = "inline-flex";
      this.elTimerVal.textContent = `${duration}s`;

      this.updateSummaryUI();
      this.populateTable();
      this.render();
    } catch (err) {
      console.error("Analysis execution error:", err);
      alert("Analysis failed: " + err.message);
    } finally {
      this.setLoading(false);
    }
  }

  setLoading(isLoading) {
    if (this.elProcessingOverlay) {
      this.elProcessingOverlay.style.display = isLoading ? "flex" : "none";
    }
    if (this.elBtnRun) {
      this.elBtnRun.disabled = isLoading;
      this.elBtnRunText.textContent = isLoading ? "Processing AI Pipeline..." : "Run AI Analysis";
    }
  }

  updateSummaryUI() {
    if (!this.analysisData || !this.analysisData.summary) return;
    const s = this.analysisData.summary;

    this.elValParcels.textContent = s.total_parcels || 0;
    this.elValBuildings.textContent = s.total_buildings || 0;
    this.elValArea.textContent = `${s.total_surveyed_area_sqm.toLocaleString()} m²`;
    this.elValBuiltup.textContent = `${s.overall_builtup_pct}%`;
    this.elBadgeParcelCount.textContent = `${s.total_parcels} Parcels`;
  }

  populateTable() {
    if (!this.analysisData || !this.analysisData.parcels) return;

    const query = (this.elParcelSearch.value || "").trim().toLowerCase();
    const statusFilter = this.elStatusFilter.value;

    let filtered = this.analysisData.parcels.filter(p => {
      const matchQuery = !query ||
        p.parcel_id.toLowerCase().includes(query) ||
        p.land_use.toLowerCase().includes(query);
      const matchStatus = statusFilter === "ALL" || p.verification_status === statusFilter;
      return matchQuery && matchStatus;
    });

    if (filtered.length === 0) {
      this.elTableBody.innerHTML = `<tr><td colspan="9" class="empty-table-msg">No matching parcels found.</td></tr>`;
      return;
    }

    let html = "";
    for (let p of filtered) {
      const isSelected = p.parcel_id === this.selectedParcelId;
      const statusClass = p.verification_status.toLowerCase().replace(/\s+/g, "-");

      html += `
        <tr class="parcel-row ${isSelected ? 'selected' : ''}" data-id="${p.parcel_id}">
          <td class="col-pid"><strong>${p.parcel_id}</strong></td>
          <td>
            <select class="table-select landuse-select" data-id="${p.parcel_id}">
              <option value="Residential" ${p.land_use === 'Residential' ? 'selected' : ''}>Residential</option>
              <option value="Commercial / Institutional" ${p.land_use.startsWith('Commercial') ? 'selected' : ''}>Commercial</option>
              <option value="Agricultural / Green Space" ${p.land_use.startsWith('Agricultural') ? 'selected' : ''}>Agricultural</option>
              <option value="Vacant / Open Plot" ${p.land_use.startsWith('Vacant') ? 'selected' : ''}>Vacant</option>
              <option value="Mixed Use" ${p.land_use === 'Mixed Use' ? 'selected' : ''}>Mixed Use</option>
            </select>
          </td>
          <td>${p.area_sqm.toLocaleString()}</td>
          <td>${p.area_gaj.toLocaleString()}</td>
          <td>${p.perimeter_m} m</td>
          <td>${p.builtup_pct}%</td>
          <td><span class="conf-badge">${Math.round(p.confidence * 100)}%</span></td>
          <td>
            <select class="table-select status-select status-${statusClass}" data-id="${p.parcel_id}">
              <option value="Verified" ${p.verification_status === 'Verified' ? 'selected' : ''}>Verified ✅</option>
              <option value="Pending Review" ${p.verification_status === 'Pending Review' ? 'selected' : ''}>Pending ⏳</option>
              <option value="Disputed" ${p.verification_status === 'Disputed' ? 'selected' : ''}>Disputed ⚠️</option>
            </select>
          </td>
          <td>
            <button class="btn-inspect" data-id="${p.parcel_id}" title="Inspect Parcel"><i class="fa-solid fa-eye"></i></button>
          </td>
        </tr>
      `;
    }

    this.elTableBody.innerHTML = html;

    // Attach Row Event Listeners
    this.elTableBody.querySelectorAll(".parcel-row").forEach(row => {
      row.addEventListener("click", (e) => {
        if (e.target.tagName === "SELECT" || e.target.tagName === "BUTTON" || e.target.closest("button")) return;
        const pid = row.getAttribute("data-id");
        this.selectParcel(pid, true);
      });
    });

    this.elTableBody.querySelectorAll(".btn-inspect").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const pid = btn.getAttribute("data-id");
        this.selectParcel(pid, true);
        this.openInspector(pid);
      });
    });

    this.elTableBody.querySelectorAll(".status-select").forEach(sel => {
      sel.addEventListener("change", async (e) => {
        const pid = sel.getAttribute("data-id");
        const newStatus = sel.value;
        await this.updateParcel(pid, { verification_status: newStatus });
      });
    });

    this.elTableBody.querySelectorAll(".landuse-select").forEach(sel => {
      sel.addEventListener("change", async (e) => {
        const pid = sel.getAttribute("data-id");
        const newLanduse = sel.value;
        await this.updateParcel(pid, { land_use: newLanduse });
      });
    });
  }

  async updateParcel(parcelId, updates) {
    if (!this.analysisData || !this.analysisData.summary) return;

    try {
      const resp = await fetch("/api/parcels/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: this.analysisData.summary.analysis_id,
          parcel_id: parcelId,
          ...updates
        })
      });

      if (!resp.ok) throw new Error("Failed to update parcel on backend");

      const result = await resp.json();
      if (result.success) {
        // Update local object
        const p = this.analysisData.parcels.find(x => x.parcel_id === parcelId);
        if (p) {
          if (updates.verification_status) p.verification_status = updates.verification_status;
          if (updates.land_use) p.land_use = updates.land_use;
        }
        this.render();
      }
    } catch (err) {
      console.error("Parcel update error:", err);
      alert("Could not update parcel: " + err.message);
    }
  }

  selectParcel(parcelId, centerOnMap = false) {
    this.selectedParcelId = parcelId;

    // Highlight row in table
    this.elTableBody.querySelectorAll(".parcel-row").forEach(row => {
      row.classList.toggle("selected", row.getAttribute("data-id") === parcelId);
    });

    if (centerOnMap) {
      const p = this.analysisData?.parcels?.find(x => x.parcel_id === parcelId);
      if (p && p.centroid_px) {
        const rect = this.container.getBoundingClientRect();
        this.panX = rect.width / 2 - p.centroid_px[0] * this.zoom;
        this.panY = rect.height / 2 - p.centroid_px[1] * this.zoom;
      }
    }

    this.openInspector(parcelId);
    this.render();
  }

  openInspector(parcelId) {
    const p = this.analysisData?.parcels?.find(x => x.parcel_id === parcelId);
    if (!p) return;

    this.elInspParcelId.textContent = `Parcel ${p.parcel_id}`;
    this.elInspectorBody.innerHTML = `
      <div class="insp-stat-row">
        <span class="insp-label">Classification:</span>
        <span class="insp-val badge-landuse">${p.land_use}</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Surveyed Area:</span>
        <span class="insp-val"><strong>${p.area_sqm.toLocaleString()} m²</strong> (${p.area_gaj.toLocaleString()} गज)</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Hectares / Acres:</span>
        <span class="insp-val">${p.area_hectares} ha / ${p.area_acres} ac</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Boundary Perimeter:</span>
        <span class="insp-val">${p.perimeter_m} meters</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Built-up Footprint:</span>
        <span class="insp-val">${p.builtup_pct}% (${p.builtup_area_sqm} m², ${p.building_count} structures)</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Boundary Compactness:</span>
        <span class="insp-val">${p.compactness} / 1.0</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">AI Model Confidence:</span>
        <span class="insp-val">${Math.round(p.confidence * 100)}%</span>
      </div>
      <div class="insp-stat-row">
        <span class="insp-label">Coordinates (Lon, Lat):</span>
        <span class="insp-val">${p.centroid_geo[0]}, ${p.centroid_geo[1]}</span>
      </div>
      <hr style="border:none;border-top:1px solid #e0e0e0;margin:15px 0;">
      <div class="insp-verify-group">
        <label><strong>Cadastral Verification Action:</strong></label>
        <div class="btn-group-status">
          <button class="btn-status-verify ${p.verification_status === 'Verified' ? 'active' : ''}" id="btnMarkVerified">
            <i class="fa-solid fa-circle-check"></i> Verify Lot
          </button>
          <button class="btn-status-dispute ${p.verification_status === 'Disputed' ? 'active' : ''}" id="btnMarkDisputed">
            <i class="fa-solid fa-triangle-exclamation"></i> Mark Disputed
          </button>
        </div>
      </div>
    `;

    document.getElementById("btnMarkVerified")?.addEventListener("click", () => {
      this.updateParcel(p.parcel_id, { verification_status: "Verified" });
      this.openInspector(p.parcel_id);
      this.populateTable();
    });

    document.getElementById("btnMarkDisputed")?.addEventListener("click", () => {
      this.updateParcel(p.parcel_id, { verification_status: "Disputed" });
      this.openInspector(p.parcel_id);
      this.populateTable();
    });

    this.elInspector.classList.add("open");
  }

  exportFile(format) {
    if (!this.analysisData || !this.analysisData.summary) {
      alert("Please run an AI analysis first before exporting data.");
      return;
    }
    const analysisId = this.analysisData.summary.analysis_id;
    window.open(`/api/export/${analysisId}/${format}`, "_blank");
  }

  // --- Rendering Loop ---
  render() {
    const rect = this.container.getBoundingClientRect();
    this.ctx.clearRect(0, 0, rect.width, rect.height);

    this.ctx.save();
    this.ctx.translate(this.panX, this.panY);
    this.ctx.scale(this.zoom, this.zoom);

    // 1. Base Orthomosaic Image or Full AI Overlay
    if (this.showOverlayImage && this.overlayImage && this.overlayImage.complete) {
      this.ctx.drawImage(this.overlayImage, 0, 0);
    } else if (this.layers.ortho && this.baseImage && this.baseImage.complete) {
      this.ctx.globalAlpha = this.layers.orthoOpacity;
      this.ctx.drawImage(this.baseImage, 0, 0);
      this.ctx.globalAlpha = 1.0;
    }

    // Render Vector Layers from geojson_pixel
    if (this.analysisData && this.analysisData.geojson_pixel && this.analysisData.geojson_pixel.features) {
      const features = this.analysisData.geojson_pixel.features;

      // 2. Road Network Lines
      if (this.layers.roads) {
        this.ctx.strokeStyle = "#ffeb3b";
        this.ctx.lineWidth = Math.max(2, 4 / this.zoom);
        this.ctx.lineCap = "round";
        for (let f of features) {
          if (f.geometry.type === "LineString") {
            this.drawLinestring(f.geometry.coordinates);
          }
        }
      }

      // 3. Cadastral Parcel Polygons
      if (this.layers.parcels) {
        for (let f of features) {
          if (f.properties?.feature_type === "parcel") {
            const isHovered = this.hoveredParcel && this.hoveredParcel.id === f.id;
            const isSelected = this.selectedParcelId === f.id;

            // Fill color
            if (isSelected) {
              this.ctx.fillStyle = "rgba(0, 229, 255, 0.45)";
            } else if (isHovered) {
              this.ctx.fillStyle = "rgba(255, 235, 59, 0.40)";
            } else {
              this.ctx.fillStyle = `rgba(33, 150, 243, ${0.18 * this.layers.parcelOpacity})`;
            }

            // Boundary stroke
            this.ctx.strokeStyle = isSelected ? "#00e5ff" : (isHovered ? "#ffeb3b" : "#0288d1");
            this.ctx.lineWidth = (isSelected || isHovered) ? Math.max(3, 4.5 / this.zoom) : Math.max(1.5, 2 / this.zoom);

            this.drawPolygon(f.geometry.coordinates);
          }
        }
      }

      // 4. Building Footprints
      if (this.layers.buildings) {
        this.ctx.fillStyle = "rgba(255, 87, 34, 0.70)";
        this.ctx.strokeStyle = "#ff3d00";
        this.ctx.lineWidth = Math.max(1.5, 2 / this.zoom);

        for (let f of features) {
          if (f.properties?.feature_type === "building") {
            this.drawPolygon(f.geometry.coordinates);
          }
        }
      }

      // 5. Centroid ID Labels
      if (this.layers.labels && this.analysisData.parcels) {
        const fontSize = Math.max(11, Math.min(16, 14 / Math.sqrt(this.zoom)));
        this.ctx.font = `600 ${fontSize}px Poppins, sans-serif`;
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";

        for (let p of this.analysisData.parcels) {
          if (p.centroid_px) {
            const [cx, cy] = p.centroid_px;
            // Draw background tag pill
            const text = p.parcel_id;
            const tw = this.ctx.measureText(text).width + 8;
            const th = fontSize + 4;

            this.ctx.fillStyle = "rgba(5, 5, 93, 0.85)";
            this.ctx.beginPath();
            this.ctx.roundRect(cx - tw / 2, cy - th / 2, tw, th, 4);
            this.ctx.fill();

            this.ctx.fillStyle = "#ffffff";
            this.ctx.fillText(text, cx, cy);
          }
        }
      }
    }

    this.ctx.restore();
  }

  drawPolygon(rings) {
    if (!rings || rings.length === 0) return;
    this.ctx.beginPath();
    for (let r = 0; r < rings.length; r++) {
      const ring = rings[r];
      if (ring.length < 3) continue;
      this.ctx.moveTo(ring[0][0], ring[0][1]);
      for (let i = 1; i < ring.length; i++) {
        this.ctx.lineTo(ring[i][0], ring[i][1]);
      }
      this.ctx.closePath();
    }
    this.ctx.fill();
    this.ctx.stroke();
  }

  drawLinestring(coords) {
    if (!coords || coords.length < 2) return;
    this.ctx.beginPath();
    this.ctx.moveTo(coords[0][0], coords[0][1]);
    for (let i = 1; i < coords.length; i++) {
      this.ctx.lineTo(coords[i][0], coords[i][1]);
    }
    this.ctx.stroke();
  }

  // Mouse / Spatial Raycasting
  screenToCanvas(screenX, screenY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = (screenX - rect.left - this.panX) / this.zoom;
    const y = (screenY - rect.top - this.panY) / this.zoom;
    return { x, y };
  }

  onMouseDown(e) {
    this.isDragging = true;
    this.dragStartX = e.clientX - this.panX;
    this.dragStartY = e.clientY - this.panY;
    this.canvas.style.cursor = "grabbing";
  }

  onMouseMove(e) {
    if (this.isDragging) {
      this.panX = e.clientX - this.dragStartX;
      this.panY = e.clientY - this.dragStartY;
      this.render();
      return;
    }

    const pt = this.screenToCanvas(e.clientX, e.clientY);
    const found = this.findParcelAt(pt.x, pt.y);

    if (found !== this.hoveredParcel) {
      this.hoveredParcel = found;
      this.render();
    }

    if (found) {
      this.canvas.style.cursor = "pointer";
      this.showTooltip(e.clientX, e.clientY, found);
    } else {
      this.canvas.style.cursor = "crosshair";
      this.hideTooltip();
    }
  }

  onMouseUp() {
    this.isDragging = false;
    this.canvas.style.cursor = "crosshair";
  }

  onMouseLeave() {
    this.isDragging = false;
    this.hoveredParcel = null;
    this.hideTooltip();
    this.render();
  }

  onWheel(e) {
    e.preventDefault();
    const rect = this.container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    this.zoomAtPoint(mouseX, mouseY, factor);
  }

  onCanvasClick(e) {
    const pt = this.screenToCanvas(e.clientX, e.clientY);
    const found = this.findParcelAt(pt.x, pt.y);
    if (found) {
      this.selectParcel(found.id, false);
    }
  }

  findParcelAt(x, y) {
    if (!this.analysisData || !this.analysisData.geojson_pixel) return null;
    const features = this.analysisData.geojson_pixel.features;

    for (let f of features) {
      if (f.properties?.feature_type === "parcel") {
        const rings = f.geometry.coordinates;
        if (rings && rings.length > 0 && this.pointInPolygon([x, y], rings[0])) {
          return f;
        }
      }
    }
    return null;
  }

  pointInPolygon(point, vs) {
    const x = point[0], y = point[1];
    let inside = false;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
      const xi = vs[i][0], yi = vs[i][1];
      const xj = vs[j][0], yj = vs[j][1];
      const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  showTooltip(screenX, screenY, feature) {
    const p = feature.properties;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = `${screenX + 15}px`;
    this.tooltip.style.top = `${screenY + 15}px`;
    this.tooltip.innerHTML = `
      <div class="tip-id">${p.parcel_id}</div>
      <div class="tip-row"><i class="fa-solid fa-vector-square"></i> Area: <strong>${p.area_sqm} m²</strong> (${p.area_gaj} गज)</div>
      <div class="tip-row"><i class="fa-solid fa-tag"></i> Land Use: ${p.land_use}</div>
      <div class="tip-row"><i class="fa-solid fa-circle-check"></i> Status: ${p.verification_status}</div>
    `;
  }

  hideTooltip() {
    this.tooltip.style.display = "none";
  }
}

// Instantiate on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.cadastralApp = new CadastralApp();
});

