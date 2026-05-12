//
// ゲームグラフィックス特論宿題アプリケーション
//
#include <cmath>

#include "GgApp.h"

// プロジェクト名
#ifndef PROJECT_NAME
#  define PROJECT_NAME "ggsample04"
#endif

// オブジェクト関連の処理
#include "object.h"

// シェーダー関連の処理
#include "shader.h"

// 変換行列関連の処理
#include "matrix.h"

// Catmull-Rom スプライン
#include "spline.h"

// アニメーションの周期（秒）
constexpr auto cycle{ 5.0 };

namespace
{
  // 四元数 (x, y, z, w), w はスカラー成分
  struct Quaternion
  {
    GLfloat x, y, z, w;
  };

  //
  // 四元数を正規化する
  //
  Quaternion normalize(const Quaternion& q)
  {
    const auto l{ std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) };

    if (l > 0.0f) return Quaternion{ q.x / l, q.y / l, q.z / l, q.w / l };

    return Quaternion{ 0.0f, 0.0f, 0.0f, 1.0f };
  }

  //
  // 宿題：軸と回転角から単位四元数を求める関数
  //
  Quaternion quaternion(float x, float y, float z, float a)
  {
    const auto l{ std::sqrt(x * x + y * y + z * z) };

    if (l <= 0.0f) return Quaternion{ 0.0f, 0.0f, 0.0f, 1.0f };

    const auto s{ std::sin(a * 0.5f) / l };

    return Quaternion{ x * s, y * s, z * s, std::cos(a * 0.5f) };
  }

  //
  // 宿題：単位四元数を球面線形補間する
  //
  Quaternion slerp(const Quaternion& q0, const Quaternion& q1, float t)
  {
    auto q{ q1 };
    auto c{ q0.x * q.x + q0.y * q.y + q0.z * q.z + q0.w * q.w };

    if (c < 0.0f)
    {
      q.x = -q.x;
      q.y = -q.y;
      q.z = -q.z;
      q.w = -q.w;
      c = -c;
    }

    if (c > 0.9995f)
    {
      return normalize(Quaternion
        {
          q0.x + (q.x - q0.x) * t,
          q0.y + (q.y - q0.y) * t,
          q0.z + (q.z - q0.z) * t,
          q0.w + (q.w - q0.w) * t
        });
    }

    const auto a{ std::acos(c) };
    const auto s{ std::sin(a) };
    const auto s0{ std::sin((1.0f - t) * a) / s };
    const auto s1{ std::sin(t * a) / s };

    return Quaternion
    {
      q0.x * s0 + q.x * s1,
      q0.y * s0 + q.y * s1,
      q0.z * s0 + q.z * s1,
      q0.w * s0 + q.w * s1
    };
  }

  //
  // 宿題：単位四元数から回転変換行列を求める
  //
  void quaternionMatrix(GLfloat* m, const Quaternion& q)
  {
    const auto x2{ q.x * q.x };
    const auto y2{ q.y * q.y };
    const auto z2{ q.z * q.z };
    const auto xy{ q.x * q.y };
    const auto yz{ q.y * q.z };
    const auto zx{ q.z * q.x };
    const auto xw{ q.x * q.w };
    const auto yw{ q.y * q.w };
    const auto zw{ q.z * q.w };

    m[ 0] = 1.0f - 2.0f * (y2 + z2);
    m[ 1] = 2.0f * (xy + zw);
    m[ 2] = 2.0f * (zx - yw);
    m[ 3] = 0.0f;

    m[ 4] = 2.0f * (xy - zw);
    m[ 5] = 1.0f - 2.0f * (z2 + x2);
    m[ 6] = 2.0f * (yz + xw);
    m[ 7] = 0.0f;

    m[ 8] = 2.0f * (zx + yw);
    m[ 9] = 2.0f * (yz - xw);
    m[10] = 1.0f - 2.0f * (x2 + y2);
    m[11] = 0.0f;

    m[12] = m[13] = m[14] = 0.0f;
    m[15] = 1.0f;
  }
}

//
// アプリケーション本体
//
int GgApp::main(int argc, const char* const* argv)
{
  // ウィンドウを作成する (この行は変更しないでください)
  Window window{ argc > 1 ? argv[1] : PROJECT_NAME };

  // 背景色を指定する
  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

  // プログラムオブジェクトの作成
  const auto program{ loadProgram(PROJECT_NAME ".vert", "pv", PROJECT_NAME ".frag", "fc") };

  // uniform 変数のインデックスの検索（見つからなければ -1）
  const auto mcLoc{ glGetUniformLocation(program, "mc") };

  // ビュー変換行列を mv に求める
  GLfloat mv[16];
  lookat(mv, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

  // 頂点属性
  static const GLfloat position[][3]
  {
    { -0.9f, -0.9f, -0.9f },  // (0)
    {  0.9f, -0.9f, -0.9f },  // (1)
    {  0.9f, -0.9f,  0.9f },  // (2)
    { -0.9f, -0.9f,  0.9f },  // (3)
    { -0.9f,  0.9f, -0.9f },  // (4)
    {  0.9f,  0.9f, -0.9f },  // (5)
    {  0.9f,  0.9f,  0.9f },  // (6)
    { -0.9f,  0.9f,  0.9f },  // (7)
  };

  // 頂点数
  constexpr auto vertices{ static_cast<GLuint>(std::size(position)) };

  // 頂点インデックス
  static const GLuint index[]
  {
    0, 1,
    1, 2,
    2, 3,
    3, 0,
    0, 4,
    1, 5,
    2, 6,
    3, 7,
    4, 5,
    5, 6,
    6, 7,
    7, 4,
  };

  // 稜線数
  constexpr auto lines{ static_cast<GLuint>(std::size(index)) };

  // 頂点配列オブジェクトの作成
  const auto vao{ createObject(vertices, position, lines, index) };

  // 平行移動の経路
  static const float route[][3]
  {
    { -2.0f, -1.0f, -3.0f },
    {  0.0f, -2.0f, -2.0f },
    { -1.0f, -1.0f, -1.0f },
    { -0.5f, -0.5f, -0.5f },
    {  0.0f,  0.0f,  0.0f },
  };

  // 通過時間 (× cycle)
  static const float transit[]
  {
    0.0f,
    0.3f,
    0.5f,
    0.7f,
    1.0f,
  };

  // 通過地点の数
  constexpr auto points{ static_cast<int>(std::size(transit)) };

  // 経過時間のリセット
  glfwSetTime(0.0);

  // ウィンドウが開いている間繰り返す
  while (window)
  {
    // ウィンドウを消去する
    glClear(GL_COLOR_BUFFER_BIT);

    // シェーダプログラムの使用開始
    glUseProgram(program);

    // 時刻の計測
    const auto t{ static_cast<float>(fmod(glfwGetTime(), cycle) / cycle) };

    // 時刻 t にもとづく回転アニメーション
    GLfloat mr[16];                   // 回転の変換行列
    // 【宿題】ここを解答してください（下の loadIdentity() を置き換えてください）
    const auto q0{ quaternion(1.0f, 0.0f, 0.0f, 1.0f) };
    const auto q1{ quaternion(0.0f, 0.0f, 1.0f, 2.0f) };
    const auto q{ slerp(q0, q1, t) };
    quaternionMatrix(mr, q);

    // 時刻 t にもとづく平行移動アニメーション
    float location[3];                // 現在位置
    spline(location, route, transit, points, t);
    GLfloat mt[16];                   // 平行移動の変換行列
    translate(mt, location[0], location[1], location[2]);

    // モデル変換行列を mm に求め，
    // それとビュー変換行列 mv の積をモデルビュー変換行列 mw に求める
    GLfloat mm[16], mw[16];
    multiply(mm, mt, mr);             // モデル変換 mm ← 移動 mt × 回転 mr
    multiply(mw, mv, mm);             // モデルビュー変換 mw ← ビュー変換 mv × モデル変換 mm

    // 透視投影変換行列を mp に求め，
    // それとモデルビュー変換行列 mw の積をクリッピング座標系への変換行列 mc に求める
    GLfloat mp[16], mc[16];
    perspective(mp, 0.5f, window.getAspect(), 1.0f, 15.0f);
    multiply(mc, mp, mw);             // クリッピング座標系への変換行列 mc ← 投影変換 mp × モデルビュー変換 mw

    // uniform 変数 mc に変換行列 mc を設定する
    glUniformMatrix4fv(mcLoc, 1, GL_FALSE, mc);

    // 描画に使う頂点配列オブジェクトの指定
    glBindVertexArray(vao);

    // 図形の描画
    glDrawElements(GL_LINES, lines, GL_UNSIGNED_INT, 0);

    // 頂点配列オブジェクトの指定解除
    glBindVertexArray(0);

    // シェーダプログラムの使用終了
    glUseProgram(0);

    // カラーバッファを入れ替えてイベントを取り出す
    window.swapBuffers();
  }

  return 0;
}
