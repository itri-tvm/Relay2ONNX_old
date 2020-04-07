/*!
 *
 * \file partition_subgraph_op.h
 * \brief Abstract class to partition graph to subgraph.
 */
#ifndef TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_
#define TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_
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
class AttrsComparor{
public:
	/*
	 * \brief Compare non default attrs
	 * \param expr the Relay expression.
	 * \return Type.
	 */
	static bool CompareNonDefault(const Attrs pself, const Attrs other);
private:
		class EqualVisitor;
		friend class EqualVisitor;
};
class AttrsComparor::EqualVisitor: public AttrVisitor {
public:
	bool result_ { true };
	EqualVisitor(const Object *lhs, const Object *rhs, const AttrsEqual &equal) :
			lhs_(lhs), rhs_(rhs), equal_(equal) {
	}
	template<typename T>
	void CompareAttr(const char *key, T *lhs_value) {
		if (!result_)
			return;
		const T *rhs_value =
				reinterpret_cast<const T*>(reinterpret_cast<const char*>(rhs_)
						+ (reinterpret_cast<const char*>(lhs_value)
								- reinterpret_cast<const char*>(lhs_)));
		if (!equal_(*lhs_value, *rhs_value)) {
			result_ = false;
		} else {
			result_ = true;
		}
	}
	void Visit(const char *key, double *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, int64_t *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, uint64_t *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, int *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, bool *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, std::string *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, void **value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, DataType *value) final {
		// do nothing
	}
	void Visit(const char *key, runtime::NDArray *value) final {
		CompareAttr(key, value);

	}
	void Visit(const char *key, runtime::ObjectRef *obj) final {
		CompareAttr(key, obj);
	}

private:
	const Object *lhs_;
	const Object *rhs_;
	const AttrsEqual &equal_;

};
class SubgraphPartitionor: public ExprMutator {
public:
	/*
	 * \brief Infer the type of expression.
	 * \param expr the Relay expression.
	 * \return Type.
	 */
	Type InferType(const Expr &expr);
	/*
	 * \brief Cast the output data to the specified data type.
	 * \param expr the Relay expression.
	 * \param tar_type the target data type.
	 * \return New expression with the cast operator.
	 */
	Expr Cast(const Expr &expr, DataType dst_type);
	/*
	 * \brief Convert op name to fused name.
	 * \param text op name
	 * \return Fused name.
	 */
	std::string ConvertOpName(std::string text);
private:
	// The packed function to make cast operator.
	const PackedFunc cast_ = (*tvm::runtime::Registry::Get("relay._make.cast"));
};
}// namespace relay
}// namespace tvm
#endif  // TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_