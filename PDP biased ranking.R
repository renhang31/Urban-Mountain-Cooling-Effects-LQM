# 2. 重新安装依赖包
install.packages(c("glue", "cli", "pillar", "R6", "withr", "gtable"))

# 3. 安装 pdp 开发版
devtools::install_github("bgreenwell/pdp")
# 安装 CRAN 最新版（可能非开发版）
install.packages("pdp")
packageVersion("pdp")
# 加载所需库
library(xgboost)
library(pdp)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(rlang)  # 用于动态变量名

# -*- coding: utf-8 -*-
# 偏依赖图（XGBoost + pdp），基于 ICE 聚合计算均值与95%区间
# 依赖：xgboost, pdp, ggplot2, ggpubr, dplyr, rlang

# ---- 1. 加载包 ----
suppressPackageStartupMessages({
  library(xgboost)
  library(pdp)
  library(ggplot2)
  library(ggpubr)
  library(dplyr)
  library(rlang)
})

# ---- 2. 主题设置 ----
theme_set(
  theme_pubr(
    base_family = "Times New Roman",
    base_size   = 18,
    legend      = "right"
  ) +
    theme(
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.line    = element_line(colour = "black", linewidth = 1.2),
      panel.grid.minor = element_blank(),
      plot.title   = element_text(hjust = 0.5, margin = margin(b = 10)),
      strip.background = element_rect(fill = "white"),
      strip.text   = element_text(face = "bold")
    )
)

# ---- 3. 颜色 ----
colors <- c("#2E75B6", "#ED7D31", "#A5A5A5", "#FFC000", "#4472C4")

# ---- 4. 读数据 ----
data <- read.csv("C:/Users/admin/Desktop/Cooling_Effects_Large_Urban_Mountains/driverFactor/2011/resampledclip_data_sorted_standardized.csv")
stopifnot("MCI" %in% names(data))
cols <- setdiff(colnames(data), "MCI")

# ---- 5. 划分训练/测试 ----
set.seed(42)
split <- sample(2, nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train <- data[split == 1, , drop = FALSE]
test  <- data[split == 2, , drop = FALSE]

# ---- 6. 训练 XGBoost ----
model <- xgboost(
  data = as.matrix(train[cols]),
  label = train$MCI,
  max_depth = 4,
  eta = 0.05,
  nrounds = 150,
  subsample = 1,
  colsample_bytree = 1,
  alpha = 0.1,
  lambda = 0.1,
  objective = "reg:squarederror",
  verbose = 0
)

# ---- 7. 预测函数（供 pdp 调用）----
pred.fun <- function(object, newdata) {
  predict(object, as.matrix(newdata))
}

# ---- 8. PDP 绘图函数（用 ICE 计算均值与95%区间）----
plot_pdp <- function(feature_name,
                     grid.resolution = 100,
                     ci = 0.95,
                     ice_sample_n = 2000) {
  stopifnot(feature_name %in% cols)
  
  # 为了控制计算量，对用于 ICE 的样本做可选抽样（不影响总体趋势）
  if (!is.null(ice_sample_n) && ice_sample_n < nrow(train)) {
    set.seed(42)
    idx <- sample(seq_len(nrow(train)), ice_sample_n)
    train_ice <- train[idx, cols, drop = FALSE]
  } else {
    train_ice <- train[cols]
  }
  
  # 生成 ICE：每个样本在特征栅格上的响应
  pd_raw <- partial(
    object = model,
    pred.var = feature_name,
    train = train_ice,
    pred.fun = pred.fun,
    grid.resolution = grid.resolution,
    ice = TRUE,
    center = FALSE,
    plot = FALSE
  )
  # pd_raw 列包含：<featureName>, yhat, yhat.id（个体编号）
  
  # 聚合为均值与分位数区间
  alpha <- (1 - ci) / 2
  pd_sum <- pd_raw %>%
    group_by(.data[[feature_name]]) %>%
    summarise(
      yhat  = mean(.data$yhat, na.rm = TRUE),
      lower = quantile(.data$yhat, probs = alpha, na.rm = TRUE, type = 7),
      upper = quantile(.data$yhat, probs = 1 - alpha, na.rm = TRUE, type = 7),
      .groups = "drop"
    )
  
  # 绘图
  ggplot(pd_sum, aes(x = .data[[feature_name]], y = yhat)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = colors[1], alpha = 0.2) +
    geom_line(color = colors[1], linewidth = 1.2) +
    labs(
      x = feature_name,
      y = "Predicted MCI",
      title = paste("Partial Dependence Plot for", feature_name)
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      axis.title = element_text(face = "bold", size = 18),
      axis.text  = element_text(size = 16),
      legend.position = "none"
    )
}

# ---- 9. 批量输出图像 ----
out_dir <- "C:/Users/admin/Desktop/Cooling_Effects_Large_Urban_Mountains/driverFactor/XGB2001"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (feature in cols) {
  p <- plot_pdp(feature)
  ggsave(
    filename = file.path(out_dir, paste0("pdp_", feature, ".png")),
    plot = p,
    device = "png",
    width = 8,
    height = 6,
    dpi = 300,
    bg = "white"
  )
}

cat("所有偏依赖图已保存至：", out_dir, "\n")

