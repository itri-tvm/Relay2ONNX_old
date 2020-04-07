#include "partition_graph_util.h"
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/nn.h>
#include "../../ir/attr_functor.h"
namespace tvm {
namespace relay {

bool AttrsComparor::CompareNonDefault(const Attrs pself,
		const Attrs other) {
	auto p = pself.operator->();
	auto o = other.operator->();
	if (!p && !o)
		return true;
	if(!(p && o))
		return false;
	if (p->type_index() != o->type_index())
		return false;
	AttrsEqual equal;
	EqualVisitor visitor(p, o, equal);
	const_cast<BaseAttrsNode*>(p)->VisitEachNonDefaultAttrs(
			&visitor);
	return visitor.result_;
}
Type SubgraphPartitionor::InferType(const Expr &expr) {
	Function func = FunctionNode::make(FreeVars(expr), expr, Type(),
			FreeTypeVars(expr, IRModule()), { });

	auto mod = IRModule::FromExpr(func);
	mod = transform::InferType()(mod);
	auto entry_func = Downcast<Function>(mod->Lookup("main"));
	Expr new_expr =
			expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
	return new_expr->checked_type();
}
Expr SubgraphPartitionor::Cast(const Expr &expr, DataType dst_type) {
	if (dst_type.is_handle())
		return expr;
	DataType src_type;
	src_type = InferType(expr).as<TensorTypeNode>()->dtype;
	if (src_type == dst_type) {
		return expr;
	} else {
		Expr new_expr =  cast_(expr, dst_type);
		return new_expr;
	}
}
std::string SubgraphPartitionor::ConvertOpName(std::string text) {
	for (size_t i = 0; i < text.length(); i++) {
		if (text[i] == '.')
			text[i] = '_';
	}
	return text;
}


}// namespace relay
}// namespace tvm
